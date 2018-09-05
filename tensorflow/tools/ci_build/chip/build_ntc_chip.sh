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

if [[ $1 == "NEON" ]]; then
 CHIP_COPTS='--copt=-march=armv7-a --copt=-mcpu=cortex-a8
 --copt=-mfloat-abi=hard --copt=-mfpu=neon
 --copt=-mthumb
 --copt=-std=gnu11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR
 --copt=-std=c++11
 --copt=-O3'
 echo "Building for CHIP, with NEON acceleration"
 WHEEL_ARCH=linux_armv7l
else
  CHIP_COPTS='--copt=-march=armv7-a --copt=-mcpu=cortex-a8
  --copt=-mfloat-abi=hard --copt=-mfpu=--copt=-mfpu=vfpv3-d16
  --copt=-mthumb
  --copt=-std=gnu11 --copt=-DS_IREAD=S_IRUSR --copt=-DS_IWRITE=S_IWUSR
  --copt=-std=c++11
  --copt=-O3'
  echo "Building for CHIP, with vfpv3 acceleration"
  WHEEL_ARCH=linux_armv7l
fi


# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

cd ${WORKSPACE_PATH}
bazel build -c opt ${CHIP_COPTS} \
  --config=monolithic \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer --cpu=armeabi \
  --crosstool_top=@local_config_arm_compiler//:toolchain \
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
