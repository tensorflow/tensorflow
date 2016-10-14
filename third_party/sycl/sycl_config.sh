#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


# A simple script to configure the SYCL tree needed for the TensorFlow OpenCL
# build. We need both COMPUTECPP toolkit $TF_OPENCL_VERSION.
# Useage:
#    * User edit sycl.config to point ComputeCPP toolkit to its local path
#    * run sycl_config.sh to generate symbolic links in the source tree to reflect
#    * the file organizations needed by TensorFlow.

print_usage() {
cat << EOF
Usage: $0 [--check]
  Configure TensorFlow's canonical view of SYCL libraries using sycl.config.
Arguments:
  --check: Only check that the proper SYCL dependencies has already been
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

source sycl.config || exit -1

OUTPUTDIR=${OUTPUTDIR:-../..}
COMPUTECPP_PATH=${COMPUTECPP_PATH:-/usr/local/computecpp}

# An error message when the SYCL toolkit is not found
function SYCLError {
  echo ERROR: $1
cat << EOF
##############################################################################
##############################################################################
SYCL $TF_OPENCL_VERSION toolkit is missing.
1. Download and install the ComputeCPP $TF_OPENCL_VERSION toolkit;
2. Run configure from the root of the source tree, before rerunning bazel;
Please refer to README.md for more details.
##############################################################################
##############################################################################
EOF
  exit -1
}

# Check that the SYCL libraries have already been properly configured in the source tree.
# We still need to create links to the gen-tree to make bazel happy.
function CheckAndLinkToSrcTree {
  ERROR_FUNC=$1
  FILE=$2
  if test ! -e $FILE; then
    $ERROR_FUNC "$PWD/$FILE cannot be found"
  fi

  # Link the output file to the source tree, avoiding self links if they are
  # the same. This could happen if invoked from the source tree by accident.
  if [ ! $($READLINK_CMD -f $PWD) == $($READLINK_CMD -f $OUTPUTDIR/third_party/sycl) ]; then
    mkdir -p $(dirname $OUTPUTDIR/third_party/sycl/$FILE)
    ln -sf $PWD/$FILE $OUTPUTDIR/third_party/sycl/$FILE
  fi
}

OSNAME=`uname -s`
if [ "$OSNAME" == "Linux" ]; then
  SYCL_LIB_PATH="lib"
  SYCL_RT_LIB_PATH="lib/libComputeCpp.so"
  SYCL_RT_LIB_STATIC_PATH="lib/libComputeCpp.a"
  READLINK_CMD="readlink"
fi

if [ "$CHECK_ONLY" == "1" ]; then
  CheckAndLinkToSrcTree SYCLError include/SYCL/sycl.h
  CheckAndLinkToSrcTree SYCLError $SYCL_RT_LIB_STATIC_PATH
  CheckAndLinkToSrcTree CudaError $SYCL_RT_LIB_PATH
  exit 0
fi

# Actually configure the source tree for TensorFlow's canonical view of SYCL
# libraries.

if test ! -e ${COMPUTECPP_PATH}/${SYCL_RT_LIB_PATH}; then
  SYCLError "cannot find ${COMPUTECPP_PATH}/${SYCL_RT_LIB_PATH}"
fi

# Helper function to build symbolic links for all files in a directory.
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

# Build links for all files in the directory, including subdirectories.
function LinkAllFiles {
  SRC_DIR=$1
  DST_DIR=$2
  find -L $SRC_DIR -type d | xargs -I {} bash -c "LinkOneDir $SRC_DIR $DST_DIR {}" || exit -1
}

# Set up the symbolic links for SYCL toolkit. We link at individual file level,
# not at the directory level.
# This is because the external library may have a different file layout from our desired structure.
mkdir -p $OUTPUTDIR/third_party/sycl
echo "Setting up SYCL include"
LinkAllFiles ${COMPUTECPP_PATH}/include $OUTPUTDIR/third_party/sycl/include || exit -1
echo "Setting up SYCL ${SYCL_LIB_PATH}"
LinkAllFiles ${COMPUTECPP_PATH}/${SYCL_LIB_PATH} $OUTPUTDIR/third_party/sycl/${SYCL_LIB_PATH} || exit -1
echo "Setting up SYCL bin"
LinkAllFiles ${COMPUTECPP_PATH}/bin $OUTPUTDIR/third_party/sycl/bin || exit -1
