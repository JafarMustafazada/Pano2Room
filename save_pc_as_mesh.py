"""
Pipeline:
  1. Load GaussianModel
  2. Reconstruct cameras from Camera_Trajectory
  3. Render RGB + depth on GPU with the project's CUDA rasterizer
  4. Fuse depth maps with Open3D TSDF
  5. Bake per-vertex colour from rendered frames
  6. Export .glb

Usage (from project root):
    python gs_to_mesh.py
    python gs_to_mesh.py --ply output/Pano2Room-results/3DGS.ply --out room.glb
    python gs_to_mesh.py --width 256 --height 256   # lower VRAM / faster
    python gs_to_mesh.py --tsdf-voxel 0.015         # finer mesh
"""

import os, sys, math, argparse, gc
import numpy as np
import torch
import open3d as o3d
import trimesh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.arguments import GSParams
from gaussian_renderer import render
from utils.graphics import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from utils.functions import rot_z_world_to_cam


# 1. Load Gaussians
def load_gaussians(ply_path: str, sh_degree: int = 1) -> GaussianModel:
    print(f"\n[1/5] Loading {ply_path} ...")
    g = GaussianModel(sh_degree=sh_degree)
    g.load_ply(ply_path)
    print(f"      {g.get_xyz.shape[0]:,} Gaussians  SH={g.active_sh_degree}")
    return g


# 2. Build cameras — mirrors pano2room.py load_camera_poses() + eval block
def load_cameras(traj_dir: str, W: int, H: int, fov_deg: float = 90.0) -> list:
    files = sorted(f for f in os.listdir(traj_dir) if f.startswith("camera_pose"))
    if not files:
        raise FileNotFoundError(f"No camera_pose_frame*.txt in {traj_dir}")
    print(f"\n[2/5] Loading {len(files)} cameras ...")

    pose_scale = 0.6
    yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    fov_x = math.radians(fov_deg)
    fov_y = focal2fov(fov2focal(fov_x, W), H)
    dummy = torch.zeros(3, H, W)  # Camera needs a dummy image

    raw = []
    for fname in files:
        vals = open(os.path.join(traj_dir, fname)).read().split()
        raw.append(np.array(vals, dtype=np.float64).reshape(4, 4))

    pano_origin = raw[0].copy()
    pano_cubemaps = raw[0].copy()
    cameras = []

    for i, pose in enumerate(raw):
        #  Same relative-pose logic as load_camera_poses()
        origin = pano_cubemaps if i < 6 else pano_origin
        rel = pose @ np.linalg.inv(origin)
        rel = np.vstack((-rel[0:1], -rel[1:2], rel[2:3], rel[3:4]))
        rel = rel @ rot_z_world_to_cam(180).cpu().numpy()
        rel[:3, 3] *= pose_scale  # w2c, scaled

        #  Same eval-camera reconstruction from run()
        ev = rel.astype(np.float32).copy()
        ev[0:2] *= -1

        Rw2c = ev[:3, :3]
        Tw2c = ev[:3, 3:]
        Rc2w = (yz_reverse @ Rw2c).T
        Tc2w = -Rc2w @ (yz_reverse @ Tw2c)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = Rc2w
        c2w[:3, 3:] = Tc2w

        #  OpenGL → COLMAP flip (loadCamerasFromData)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T  # transposed for glm
        T = w2c[:3, 3]

        cam = Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=fov_x,
            FoVy=fov_y,
            image=dummy,
            gt_alpha_mask=None,
            image_name=files[i],
            uid=i,
            data_device="cuda",
        )
        cameras.append(cam)

    print(f"      {len(cameras)} cameras  {W}×{H}  FoV={fov_deg}°")
    return cameras


# 3. Render all cameras on GPU
def render_all(gaussians: GaussianModel, cameras: list, white_bg: bool = False) -> list:
    print(f"\n[3/5] Rendering {len(cameras)} views on GPU ...")
    opt = GSParams()
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    frames = []
    with torch.no_grad():
        for i, cam in enumerate(cameras):
            out = render(cam, gaussians, opt, bg, render_only=True)
            rgb = (
                out["render"]
                .squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            depth = out["depth"].squeeze().cpu().numpy().astype(np.float32)
            frames.append({"rgb": rgb, "depth": depth, "cam": cam})
            if (i + 1) % 10 == 0 or (i + 1) == len(cameras):
                print(f"      {i+1}/{len(cameras)}", end="\r", flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    print(
        f"\n      Done. Example depth range: "
        f"[{frames[0]['depth'].min():.3f}, {frames[0]['depth'].max():.3f}] m"
    )
    return frames


# 4. TSDF fusion — CPU-side but memory-efficient (~100 MB for 141 cameras)
def tsdf_fuse(
    frames: list, voxel_size: float = 0.02, trunc: float = 0.08
) -> o3d.geometry.TriangleMesh:
    print(
        f"\n[4/5] TSDF fusion  voxel={voxel_size*100:.1f}cm  trunc={trunc*100:.1f}cm ..."
    )
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, fr in enumerate(frames):
        cam = fr["cam"]
        H, W = fr["rgb"].shape[:2]

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W / 2.0, H / 2.0)

        # world_view_transform is the transposed w2c (glm convention).
        # Open3D expects row-major w2c → transpose back.
        extr = cam.world_view_transform.cpu().numpy().T  # (4,4) standard w2c

        # Depth: rasterizer returns metres. TSDF integrate() needs
        # depth_scale=1 when depth is already in metres (float32 image).
        depth_f32 = np.ascontiguousarray(fr["depth"].astype(np.float32))
        # Zero-out pixels where depth is 0 (no Gaussian hit) so TSDF ignores them
        depth_f32[depth_f32 <= 0] = 0.0

        color_u8 = np.ascontiguousarray((fr["rgb"] * 255).clip(0, 255).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_u8),
            o3d.geometry.Image(depth_f32),
            depth_scale=1.0,  # depth already in metres
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        vol.integrate(rgbd, intr, extr)

        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            print(f"      integrated {i+1}/{len(frames)}", end="\r", flush=True)

    print("\n      Extracting surface ...")
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    V = len(np.asarray(mesh.vertices))
    F = len(np.asarray(mesh.triangles))
    print(f"      {V:,} vertices   {F:,} triangles")
    return mesh


# 5. Bake per-vertex colour from rendered frames
def bake_colors(
    mesh: o3d.geometry.TriangleMesh, frames: list
) -> o3d.geometry.TriangleMesh:
    print("\n[5/5] Baking vertex colours ...")

    verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V, 3)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)  # (V, 3)
    V = len(verts)
    color_sum = np.zeros((V, 3), dtype=np.float64)
    weight_sum = np.zeros(V, dtype=np.float64)

    for fr in frames:
        cam = fr["cam"]
        rgb = fr["rgb"]  # (H, W, 3)  float32 [0,1]
        depth = fr["depth"]  # (H, W)     float32 metres
        H, W = rgb.shape[:2]

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)

        # Standard row-major w2c
        extr = cam.world_view_transform.cpu().numpy().T  # (4,4)
        R_wc = extr[:3, :3].astype(np.float32)
        t_wc = extr[:3, 3].astype(np.float32)

        # Camera position in world space
        cam_pos = np.asarray(cam.camera_center.cpu(), dtype=np.float32)

        # Project vertices
        v_cam = (R_wc @ verts.T).T + t_wc[None, :]  # (V, 3)
        z = v_cam[:, 2]  # (V,)

        u = v_cam[:, 0] * fx / (z + 1e-8) + W / 2.0
        v_ = v_cam[:, 1] * fy / (z + 1e-8) + H / 2.0
        pu = np.round(u).astype(np.int32)
        pv = np.round(v_).astype(np.int32)

        # Frustum mask
        ok = (z > 0.01) & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
        if not ok.any():
            continue

        # Depth consistency (10 cm tolerance)
        pu_c = pu.clip(0, W - 1)
        pv_c = pv.clip(0, H - 1)
        rd = depth[pv_c, pu_c]
        ok &= (np.abs(rd - z) < 0.10) & (rd > 0)

        # View-angle weight
        vray = cam_pos[None, :] - verts  # (V, 3)
        vray_n = vray / (np.linalg.norm(vray, axis=1, keepdims=True) + 1e-8)
        cos_ang = (vray_n * normals).sum(axis=1)
        ok &= cos_ang > 0.1

        if not ok.any():
            continue

        w = cos_ang[ok]
        color_sum[ok] += rgb[pv_c[ok], pu_c[ok]] * w[:, None]
        weight_sum[ok] += w

    # Normalise
    w_safe = np.maximum(weight_sum, 1e-8)[:, None]
    baked = np.clip(color_sum / w_safe, 0.0, 1.0)

    # Fallback for any unvisited vertex: use TSDF colour
    if mesh.has_vertex_colors():
        tsdf_c = np.asarray(mesh.vertex_colors, dtype=np.float32)
        unvisited = weight_sum < 1e-6
        baked[unvisited] = tsdf_c[unvisited]
        n_unvisited = unvisited.sum()
        if n_unvisited:
            print(f"      {n_unvisited:,} unvisited vertices → TSDF colour fallback")

    mesh.vertex_colors = o3d.utility.Vector3dVector(baked.astype(np.float64))
    return mesh


def export_glb(
    mesh: o3d.geometry.TriangleMesh, out_path: str, target_faces: int | None = None
):
    if target_faces:
        n = len(np.asarray(mesh.triangles))
        if n > target_faces:
            print(f"\n      Decimating {n:,} → {target_faces:,} faces ...")
            mesh = mesh.simplify_quadric_decimation(target_faces)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)

    rgba = np.ones((len(verts), 4), dtype=np.uint8) * 255
    rgba[:, :3] = (colors * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=rgba, process=False
    ).export(out_path)

    mb = os.path.getsize(out_path) / 1e6
    print(
        f"\n  ✓  {out_path}  ({len(verts):,} verts  {len(faces):,} tris  {mb:.1f} MB)"
    )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("ply", nargs="?", default="output/6_Pano2Room-results/3DGS.ply")
    p.add_argument("--out", default="output/6_Pano2Room-results/mesh.glb")
    p.add_argument("--cameras", default="input/Camera_Trajectory")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--sh-degree", type=int, default=1, dest="sh_degree")
    p.add_argument("--white-bg", action="store_true", dest="white_bg")
    p.add_argument("--tsdf-voxel", type=float, default=0.02, dest="voxel")
    p.add_argument("--tsdf-trunc", type=float, default=0.08, dest="trunc")
    p.add_argument("--target-faces", type=int, default=None, dest="faces")
    args = p.parse_args()

    print("=" * 55)
    print("  3DGS → GLB  (CUDA render + TSDF fusion)")
    print("=" * 55)
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")

    g = load_gaussians(args.ply, args.sh_degree)
    cs = load_cameras(args.cameras, args.width, args.height, args.fov)
    fs = render_all(g, cs, args.white_bg)

    # Free Gaussian GPU memory before TSDF (not needed anymore)
    del g
    torch.cuda.empty_cache()
    gc.collect()

    mesh = tsdf_fuse(fs, voxel_size=args.voxel, trunc=args.trunc)
    mesh = bake_colors(mesh, fs)
    export_glb(mesh, args.out, target_faces=args.faces)


if __name__ == "__main__":
    main()
