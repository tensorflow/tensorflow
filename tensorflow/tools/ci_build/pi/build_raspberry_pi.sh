#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
set -e

# By default this builds packages for the Pi Two and Three only, since the NEON support
# this allows makes calculations many times faster. To support the Pi One or Zero, pass
# PI_ONE as the first argument to the script, for example:
# tensorflow/tools/ci_build/pi/build_raspberry_pi.sh PI_ONE
#
# To install the cross-compilation support for Python this script needs on Ubuntu Trusty, run
# something like these steps, after backing up your original /etc/apt/sources.list file:
#
# dpkg --add-architecture armhf
# echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
# echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
# echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
# echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
# sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
# apt-get update
# apt-get install -y libpython-all-dev:armhf
#
# Make sure you have an up to date version of the Bazel build tool installed too.

export TF_ENABLE_XLA=0

yes '' | ./configure

# Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
sudo sed -i 's/define CURL_SIZEOF_LONG 8/define CURL_SIZEOF_LONG 4/g' /usr/include/curl/curlbuild.h
sudo sed -i 's/define CURL_SIZEOF_CURL_OFF_T 8/define CURL_SIZEOF_CURL_OFF_T 4/g' /usr/include/curl/curlbuild.h

# The system-installed OpenSSL headers get pulled in by the latest BoringSSL
# release on this configuration, so move them before we build:
if [ -d /usr/include/openssl ]; then
  sudo mv /usr/include/openssl /usr/include/openssl.original
fi

WORKSPACE_PATH=`pwd`

# Build the OpenBLAS library, which is faster than Eigen on the Pi Zero/One.
# TODO(petewarden) - It would be nicer to move this into the main Bazel build
# process if we can maintain a build file for this.
TOOLCHAIN_INSTALL_PATH=/tmp/toolchain_install/
sudo rm -rf ${TOOLCHAIN_INSTALL_PATH}
mkdir ${TOOLCHAIN_INSTALL_PATH}
cd ${TOOLCHAIN_INSTALL_PATH}
curl -L https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz -o toolchain.tar.gz
tar xzf toolchain.tar.gz
mv rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5 tools

CROSSTOOL_CC=${TOOLCHAIN_INSTALL_PATH}/tools/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/bin/arm-rpi-linux-gnueabihf-gcc

OPENBLAS_SRC_PATH=/tmp/openblas_src/
sudo rm -rf ${OPENBLAS_SRC_PATH}
git clone https://github.com/xianyi/OpenBLAS ${OPENBLAS_SRC_PATH}
cd ${OPENBLAS_SRC_PATH}
# The commit after this introduced Fortran compile issues. In theory they should
# be solvable using NOFORTRAN=1 on the make command, but my initial tries didn't
# work, so pinning to the last know good version.
git checkout 5a6a2bed9aff0ba8a18651d5514d029c8cae336a
# If this path is changed, you'll also need to update
# cxx_builtin_include_directory in third_party/toolchains/cpus/arm/CROSSTOOL.tpl
OPENBLAS_INSTALL_PATH=/tmp/openblas_install/
make CC=${CROSSTOOL_CC} FC=${CROSSTOOL_CC} HOSTCC=gcc TARGET=ARMV6
make PREFIX=${OPENBLAS_INSTALL_PATH} install

if [[ $1 == "PI_ONE" ]]; then
  PI_COPTS="--copt=-march=armv6 --copt=-mfpu=vfp
  --copt=-DUSE_GEMM_FOR_CONV --copt=-DUSE_OPENBLAS
  --copt=-isystem --copt=${OPENBLAS_INSTALL_PATH}/include/
  --copt=-std=gnu11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR
  --copt=-fpermissive
  --linkopt=-L${OPENBLAS_INSTALL_PATH}/lib/
  --linkopt=-l:libopenblas.a"
  echo "Building for the Pi One/Zero, with no NEON support"
  WHEEL_ARCH=linux_armv6l
else
  PI_COPTS="--copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
  --copt=-std=gnu11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR
  --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
  --define=raspberry_pi_with_neon=true"
  WHEEL_ARCH=linux_armv7l
  echo "Building for the Pi Two/Three, with NEON acceleration"
fi

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

cd ${WORKSPACE_PATH}
bazel build -c opt ${PI_COPTS} \
  --config=monolithic \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer --cpu=armeabi \
  --crosstool_top=@local_config_arm_compiler//:toolchain \
  --define tensorflow_mkldnn_contraction_kernel=0 \
  --verbose_failures \
  //tensorflow:libtensorflow.so \
  //tensorflow:libtensorflow_framework.so \
  //tensorflow/tools/benchmark:benchmark_model \
  //tensorflow/tools/pip_package:build_pip_package

OUTDIR=output-artifacts
mkdir -p "${OUTDIR}"
echo "Final outputs will go to ${OUTDIR}"

# Build a universal wheel.
BDIST_OPTS="--universal" \
  bazel-bin/tensorflow/tools/pip_package/build_pip_package "${OUTDIR}"

OLD_FN=$(ls "${OUTDIR}" | grep -m 1 \.whl)
SUB='s/tensorflow-([^-]+)-([^-]+)-.*/tensorflow-\1-\2-none-'${WHEEL_ARCH}'.whl/; print'
NEW_FN=$(echo "${OLD_FN}" | perl -ne "${SUB}")
mv "${OUTDIR}/${OLD_FN}" "${OUTDIR}/${NEW_FN}"
cp bazel-bin/tensorflow/tools/benchmark/benchmark_model "${OUTDIR}"
cp bazel-bin/tensorflow/libtensorflow.so "${OUTDIR}"
cp bazel-bin/tensorflow/libtensorflow_framework.so "${OUTDIR}"

echo "Output can be found here:"
find "${OUTDIR}"
