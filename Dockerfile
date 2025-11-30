# Use CUDA base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PIP_BREAK_SYSTEM_PACKAGES=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3 python3-pip python3-venv curl zip git git-lfs wget vim libgl1 libglib2.0-0 \
   python3-dev build-essential gcc \
   && ln -sf /usr/bin/python3 /usr/bin/python \
   && ln -sf /usr/bin/pip3 /usr/bin/pip \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir gdown jupyterlab jupyterlab-lsp \
    jupyter-server jupyter-server-terminals \
    ipykernel jupyterlab_code_formatter huggingface_hub[cli] \
    ninja packaging

# Create the final image
FROM base AS final

# Clone the repository in the final stage
RUN pip install torch torchvision torchaudio
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe
# Install requirements but exclude flash-attn to avoid build issues
RUN grep -v -i "flash-attn\|flash-attention" /diffusion_pipe/requirements.txt > /tmp/requirements_no_flash.txt && \
    pip install -r /tmp/requirements_no_flash.txt


COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
CMD ["/start_script.sh"]