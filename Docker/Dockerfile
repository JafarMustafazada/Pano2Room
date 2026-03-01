# STAGE 1: Builder - contains full build toolchain and builds Python packages
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniforge3
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV CUDA_ROOT=/usr/local/cuda-12.1
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

## RTX 3070 TI
ENV TCNN_CUDA_ARCHITECTURES=86
ENV TORCH_CUDA_ARCH_LIST="8.6"
##

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Install base build/runtime packages needed for building wheels and native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    build-essential \
    git \
    unzip \
    cmake \
    ninja-build \
    pkg-config \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda) into CONDA_DIR
RUN curl -L -o /tmp/Miniforge3.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/Miniforge3.sh -b -p ${CONDA_DIR} \
    && rm /tmp/Miniforge3.sh

# Create the conda environment and upgrade pip/setuptools/wheel
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda create -y -n menv2 python=3.12 \
    && conda clean -afy \
    && . ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate menv2 \
    && pip install --no-cache-dir -U pip setuptools wheel

# Install pytorch3d via conda
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate menv2 \
    && conda install -y -c pytorch3d -c pytorch -c nvidia -c conda-forge \
            "pytorch3d=0.7.8=py312_cu121_pyt241" "mkl<2024.1" gxx_linux-64=12 || true


COPY ./Pano2Room/requirements.txt /workspace/Pano2Room/requirements.txt

# uncomment for local installing in case via github does not work
# COPY ./tiny-cuda-nn /workspace/tiny-cuda-nn
# COPY ./detectron2 /workspace/detectron2
# COPY ./simple-knn /workspace/simple-knn
# COPY ./diff-gaussian-rasterization-w-depth /workspace/diff-gaussian-rasterization-w-depth

# Install Pano2Room requirements first (inside the conda env)
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate menv2 \
    && if [ -f /workspace/Pano2Room/requirements.txt ]; then \
         echo "Installing Pano2Room requirements..." && \
         pip --no-cache-dir install -r /workspace/Pano2Room/requirements.txt || \
         (echo "WARNING: pip install -r Pano2Room/requirements.txt failed"; exit 1) ; \
       else \
         echo "No Pano2Room/requirements.txt found — skipping."; \
       fi

# BUILD & INSTALL LOCAL PACKAGES INTO THE ENV
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate menv2 \
    && set -x \
    # && (pip --no-cache-dir install /workspace/tiny-cuda-nn/bindings/torch --no-build-isolation || echo "!! tiny-cuda-nn build failed") \
    # && (pip --no-cache-dir install /workspace/simple-knn --no-build-isolation || echo "!! simple-knn build failed") \
    # && (pip --no-cache-dir install /workspace/diff-gaussian-rasterization-w-depth --no-build-isolation || echo "!! diff-gauss build failed") \
    # && (pip --no-cache-dir install /workspace/detectron2 --no-build-isolation || echo "!! detectron2 build failed")
    && (pip --no-cache-dir install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation || echo "!! detectron2 build failed") \
    && (pip --no-cache-dir install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation || echo "!! detectron2 build failed") \
    && (pip --no-cache-dir install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation || echo "!! detectron2 build failed") \
    && (pip --no-cache-dir install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation || echo "!! detectron2 build failed")

# Clean up conda caches and package caches to reduce final size
RUN . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate menv2 \
    && conda clean -afy \
    && rm -rf /root/.cache/pip

# STAGE 2: Runtime - small runtime image with pre-built conda env copied in
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniforge3
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV CUDA_ROOT=/usr/local/cuda-12.1
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
ENV TCNN_CUDA_ARCHITECTURES=86
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Minimal runtime packages required (plus gosu/tini for signal handling and privilege drop)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bash \
    tar \
    gzip \
    libgl1 \
    libglib2.0-0 \
    gosu \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root runtime user
ARG USERNAME=pano
ARG HOME_DIR=/home/${USERNAME}
RUN groupadd -r ${USERNAME} && useradd -m -r -g ${USERNAME} -s /bin/bash ${USERNAME} \
    && mkdir -p ${HOME_DIR}/workspace \
    && chown -R ${USERNAME}:${USERNAME} ${HOME_DIR}

# Copy the conda installation (including envs) from the builder
COPY --from=builder --chown=${USERNAME}:${USERNAME} ${CONDA_DIR} ${CONDA_DIR}

# Create an entrypoint script that activates the env and runs the command as the runtime user.
COPY --chown=${USERNAME}:${USERNAME} docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh
RUN chmod +x /usr/local/bin/docker_entrypoint.sh || true

# Copy optional default main script into image (if you use one)
COPY --chown=${USERNAME}:${USERNAME} run_pano2room_inner.sh /usr/local/bin/run_pano2room_inner.sh
RUN chmod +x /usr/local/bin/run_pano2room_inner.sh || true

# Prepare workspace and switch to non-root user
RUN mkdir -p /home/${USERNAME}/workspace && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/workspace
USER ${USERNAME}
ENV HOME=${HOME_DIR}
WORKDIR ${HOME_DIR}/workspace

# Runs scripts/run_Pano2Room.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker_entrypoint.sh"]