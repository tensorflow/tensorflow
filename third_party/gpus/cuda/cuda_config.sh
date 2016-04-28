#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# A simple script to configure the Cuda tree needed for the TensorFlow GPU
# build. We need both Cuda toolkit $TF_CUDA_VERSION and Cudnn $TF_CUDNN_VERSION.
# Useage:
#    * User edit cuda.config to point both Cuda toolkit and Cudnn libraries to their local path
#    * run cuda_config.sh to generate symbolic links in the source tree to reflect
#    * the file organizations needed by TensorFlow.

print_usage() {
cat << EOF
Usage: $0 [--check]
  Configure TensorFlow's canonical view of Cuda libraries using cuda.config.
Arguments:
  --check: Only check that the proper Cuda dependencies has already been
       properly configured in the source tree. It also creates symbolic links to
       the files in the gen-tree to make bazel happy.
EOF
}

CHECK_ONLY=0
# Parse the arguments. Add more arguments as the "case" line when needed.
while [[ $# -gt 0 ]]; do
  argument="$1"
  shift
  case $argument in
    --check)
      CHECK_ONLY=1
      ;;
    *)
      echo "Error: unknown arguments"
      print_usage
      exit -1
      ;;
  esac
done

source cuda.config || exit -1

OUTPUTDIR=${OUTPUTDIR:-../../..}
CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
CUDNN_INSTALL_BASEDIR=${CUDNN_INSTALL_PATH:-/usr/local/cuda}

if [[ -z "$TF_CUDA_VERSION" ]]; then
  TF_CUDA_EXT=""
else
  TF_CUDA_EXT=".$TF_CUDA_VERSION"
fi

if [[ -z "$TF_CUDNN_VERSION" ]]; then
  TF_CUDNN_EXT=""
else
  TF_CUDNN_EXT=".$TF_CUDNN_VERSION"
fi

# An error message when the Cuda toolkit is not found
function CudaError {
  echo ERROR: $1
cat << EOF
##############################################################################
##############################################################################
Cuda $TF_CUDA_VERSION toolkit is missing.
1. Download and install the CUDA $TF_CUDA_VERSION toolkit and CUDNN $TF_CUDNN_VERSION library;
2. Run configure from the root of the source tree, before rerunning bazel;
Please refer to README.md for more details.
##############################################################################
##############################################################################
EOF
  exit -1
}

# An error message when CUDNN is not found
function CudnnError {
  echo ERROR: $1
cat << EOF
##############################################################################
##############################################################################
Cudnn $TF_CUDNN_VERSION is missing.
1. Download and install the CUDA $TF_CUDA_VERSION toolkit and CUDNN $TF_CUDNN_VERSION library;
2. Run configure from the root of the source tree, before rerunning bazel;
Please refer to README.md for more details.
##############################################################################
##############################################################################
EOF
  exit -1
}

# Check that Cuda libraries has already been properly configured in the source tree.
# We still need to create links to the gen-tree to make bazel happy.
function CheckAndLinkToSrcTree {
  ERROR_FUNC=$1
  FILE=$2
  if test ! -e $FILE; then
    $ERROR_FUNC "$PWD/$FILE cannot be found"
  fi

  # Link the output file to the source tree, avoiding self links if they are
  # the same. This could happen if invoked from the source tree by accident.
  if [ ! $($READLINK_CMD -f $PWD) == $($READLINK_CMD -f $OUTPUTDIR/third_party/gpus/cuda) ]; then
    mkdir -p $(dirname $OUTPUTDIR/third_party/gpus/cuda/$FILE)
    ln -sf $PWD/$FILE $OUTPUTDIR/third_party/gpus/cuda/$FILE
  fi
}

OSNAME=`uname -s`
if [ "$OSNAME" == "Linux" ]; then
  CUDA_LIB_PATH="lib64"
  CUDA_CUPTI_LIB_DIR="extras/CUPTI/lib64"
  CUDA_RT_LIB_PATH="lib64/libcudart.so${TF_CUDA_EXT}"
  CUDA_RT_LIB_STATIC_PATH="lib64/libcudart_static.a"
  CUDA_BLAS_LIB_PATH="lib64/libcublas.so${TF_CUDA_EXT}"
  CUDA_DNN_LIB_PATH="lib64/libcudnn.so${TF_CUDNN_EXT}"
  CUDA_DNN_LIB_ALT_PATH="libcudnn.so${TF_CUDNN_EXT}"
  CUDA_FFT_LIB_PATH="lib64/libcufft.so${TF_CUDA_EXT}"
  CUDA_CUPTI_LIB_PATH="extras/CUPTI/lib64/libcupti.so${TF_CUDA_EXT}"
  READLINK_CMD="readlink"
elif [ "$OSNAME" == "Darwin" ]; then
  CUDA_LIB_PATH="lib"
  CUDA_CUPTI_LIB_DIR="extras/CUPTI/lib"
  CUDA_RT_LIB_PATH="lib/libcudart${TF_CUDA_EXT}.dylib"
  CUDA_RT_LIB_STATIC_PATH="lib/libcudart_static.a"
  CUDA_BLAS_LIB_PATH="lib/libcublas${TF_CUDA_EXT}.dylib"
  CUDA_DNN_LIB_PATH="lib/libcudnn${TF_CUDNN_EXT}.dylib"
  CUDA_DNN_LIB_ALT_PATH="libcudnn${TF_CUDNN_EXT}.dylib"
  CUDA_FFT_LIB_PATH="lib/libcufft${TF_CUDA_EXT}.dylib"
  CUDA_CUPTI_LIB_PATH="extras/CUPTI/lib/libcupti${TF_CUDA_EXT}.dylib"
  READLINK_CMD="greadlink"
fi

if [ "$CHECK_ONLY" == "1" ]; then
  CheckAndLinkToSrcTree CudaError include/cuda.h
  CheckAndLinkToSrcTree CudaError include/cublas.h
  CheckAndLinkToSrcTree CudnnError include/cudnn.h
  CheckAndLinkToSrcTree CudaError extras/CUPTI/include/cupti.h
  CheckAndLinkToSrcTree CudaError $CUDA_RT_LIB_STATIC_PATH
  CheckAndLinkToSrcTree CudaError $CUDA_BLAS_LIB_PATH
  CheckAndLinkToSrcTree CudnnError $CUDA_DNN_LIB_PATH
  CheckAndLinkToSrcTree CudaError $CUDA_RT_LIB_PATH
  CheckAndLinkToSrcTree CudaError $CUDA_FFT_LIB_PATH
  CheckAndLinkToSrcTree CudaError $CUDA_CUPTI_LIB_PATH
  exit 0
fi

# Actually configure the source tree for TensorFlow's canonical view of Cuda
# libraries.

if test ! -e ${CUDA_TOOLKIT_PATH}/${CUDA_RT_LIB_PATH}; then
  CudaError "cannot find ${CUDA_TOOLKIT_PATH}/${CUDA_RT_LIB_PATH}"
fi

if test ! -e ${CUDA_TOOLKIT_PATH}/${CUDA_CUPTI_LIB_PATH}; then
  CudaError "cannot find ${CUDA_TOOLKIT_PATH}/${CUDA_CUPTI_LIB_PATH}"
fi

if test ! -d ${CUDNN_INSTALL_BASEDIR}; then
  CudnnError "cannot find dir: ${CUDNN_INSTALL_BASEDIR}"
fi

# Locate cudnn.h
if test -e ${CUDNN_INSTALL_BASEDIR}/cudnn.h; then
  CUDNN_HEADER_DIR=${CUDNN_INSTALL_BASEDIR}
elif test -e ${CUDNN_INSTALL_BASEDIR}/include/cudnn.h; then
  CUDNN_HEADER_DIR=${CUDNN_INSTALL_BASEDIR}/include
elif test -e /usr/include/cudnn.h; then
  CUDNN_HEADER_DIR=/usr/include
else
  CudnnError "cannot find cudnn.h under: ${CUDNN_INSTALL_BASEDIR}"
fi

# Locate libcudnn
if test -e ${CUDNN_INSTALL_BASEDIR}/${CUDA_DNN_LIB_PATH}; then
  CUDNN_LIB_INSTALL_PATH=${CUDNN_INSTALL_BASEDIR}/${CUDA_DNN_LIB_PATH}
elif test -e ${CUDNN_INSTALL_BASEDIR}/${CUDA_DNN_LIB_ALT_PATH}; then
  CUDNN_LIB_INSTALL_PATH=${CUDNN_INSTALL_BASEDIR}/${CUDA_DNN_LIB_ALT_PATH}
else
  CudnnError "cannot find ${CUDA_DNN_LIB_PATH} or ${CUDA_DNN_LIB_ALT_PATH} under: ${CUDNN_INSTALL_BASEDIR}"
fi

# Helper function to build symbolic links for all files under a directory.
function LinkOneDir {
  SRC_PREFIX=$1
  DST_PREFIX=$2
  SRC_DIR=$3
  DST_DIR=$(echo $SRC_DIR | sed "s,^$SRC_PREFIX,$DST_PREFIX,")
  mkdir -p $DST_DIR
  FILE_LIST=$(find -L $SRC_DIR -maxdepth 1 -type f)
  if test "$FILE_LIST" != ""; then
    ln -sf $FILE_LIST $DST_DIR/ || exit -1
  fi
}
export -f LinkOneDir

# Build links for all files under the directory, including subdirectoreis.
function LinkAllFiles {
  SRC_DIR=$1
  DST_DIR=$2
  find -L $SRC_DIR -type d | xargs -I {} bash -c "LinkOneDir $SRC_DIR $DST_DIR {}" || exit -1
}

# Set up the symbolic links for cuda toolkit. We link at individual file level,
# not at the directory level.
# This is because the external library may have different file layout from our desired structure.
mkdir -p $OUTPUTDIR/third_party/gpus/cuda
echo "Setting up Cuda include"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/include $OUTPUTDIR/third_party/gpus/cuda/include || exit -1
echo "Setting up Cuda ${CUDA_LIB_PATH}"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/${CUDA_LIB_PATH} $OUTPUTDIR/third_party/gpus/cuda/${CUDA_LIB_PATH} || exit -1
echo "Setting up Cuda bin"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/bin $OUTPUTDIR/third_party/gpus/cuda/bin || exit -1
echo "Setting up Cuda nvvm"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/nvvm $OUTPUTDIR/third_party/gpus/cuda/nvvm || exit -1
echo "Setting up CUPTI include"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/extras/CUPTI/include $OUTPUTDIR/third_party/gpus/cuda/extras/CUPTI/include || exit -1
echo "Setting up CUPTI lib64"
LinkAllFiles ${CUDA_TOOLKIT_PATH}/${CUDA_CUPTI_LIB_DIR} $OUTPUTDIR/third_party/gpus/cuda/${CUDA_CUPTI_LIB_DIR} || exit -1

# Set up symbolic link for cudnn
ln -sf $CUDNN_HEADER_DIR/cudnn.h $OUTPUTDIR/third_party/gpus/cuda/include/cudnn.h || exit -1
ln -sf $CUDNN_LIB_INSTALL_PATH $OUTPUTDIR/third_party/gpus/cuda/$CUDA_DNN_LIB_PATH || exit -1
