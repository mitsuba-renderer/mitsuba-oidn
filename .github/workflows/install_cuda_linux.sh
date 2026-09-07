#!/bin/bash
# Installs the parts of the CUDA toolkit needed to build the OIDN CUDA device
# inside the manylinux_2_28 (AlmaLinux 8) container. Only x86_64 is supported.
set -euo pipefail

if [ "$(uname -m)" != "x86_64" ]; then
  echo "Skipping CUDA on $(uname -m)"
  exit 0
fi

dnf install -y dnf-plugins-core
dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-12-8 cuda-cudart-devel-12-8 cuda-driver-devel-12-8
