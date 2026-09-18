#!/bin/bash
# setup_uv.sh - load modules and project environment

# Set default permissions (files are group-readable)
umask 0006

# Load required modules
module load fsl
module load python/3.12.1
module load gcc/12.4.0
module load uv

export UV_PYTHON=$(which python3)

export UV_CACHE_DIR="/oak/stanford/groups/russpold/uv_cache_pool/uv_cache_YOURSUNETID"