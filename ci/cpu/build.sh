#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuDF CPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Install Openjdk"
conda install -c anaconda openjdk

logger "Install maven"
conda install --no-deps -c conda-forge maven

logger "Check versions..."
python --version
gcc --version
g++ --version
java -version
mvn -version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# INSTALL - Install NVIDIA driver
################################################################################

logger "Install NVIDIA driver for CUDA $CUDA..."
apt-get update -q
DRIVER_VER="396.44-1"
LIBCUDA_VER="396"
if [ "$CUDA" == "10.0" ]; then
  DRIVER_VER="410.72-1"
  LIBCUDA_VER="410"
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cuda-drivers=${DRIVER_VER} libcuda1-${LIBCUDA_VER}

################################################################################
# BUILD - Conda package builds (conda deps: libcudf <- libcudf_cffi <- cudf)
################################################################################

logger "Build conda pkg for libcudf..."
source ci/cpu/libcudf/build_libcudf.sh

logger "Build conda pkg for cudf..."
source ci/cpu/cudf/build_cudf.sh

################################################################################
# BUILD - libcudfjni
################################################################################

logger "Build cudfjni"
conda create -n java-cudf --clone gdf
conda activate java-cudf
conda install -c /conda/envs/gdf/conda-bld -y cudf=$MINOR_VERSION
cd $WORKSPACE/java
mvn -Dmaven.repo.local=$WORKSPACE/.m2 clean install -DskipTests
conda deactivate

################################################################################
# UPLOAD - Conda packages
################################################################################
cd $WORKSPACE
logger "Upload conda pkgs..."
source ci/cpu/upload_anaconda.sh
