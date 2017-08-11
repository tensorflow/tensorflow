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

yes '' | ./configure

# We need to update the Eigen version, because of compiler failures on ARM when
# using the version currently (Aug 10th 2017) pulled by mainline TensorFlow. We
# should be able to get rid of this hack once
# https://github.com/tensorflow/tensorflow/issues/9697 is addressed.
sed -i 's/f3a22f35b044/d781c1de9834/g' tensorflow/workspace.bzl
sed -i 's/ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4/a34b208da6ec18fa8da963369e166e4a368612c14d956dd2f9d7072904675d9b/g' tensorflow/workspace.bzl

# Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
sudo sed -i 's/define CURL_SIZEOF_LONG 8/define CURL_SIZEOF_LONG 4/g' /usr/include/curl/curlbuild.h
sudo sed -i 's/define CURL_SIZEOF_CURL_OFF_T 8/define CURL_SIZEOF_CURL_OFF_T 4/g' /usr/include/curl/curlbuild.h

if [[ $1 == "PI_ONE" ]]; then
  PI_COPTS="--copt=-march=armv6 --copt=-mfpu=vfp"
  echo "Building for the Pi One/Zero, with no NEON support"
else
  PI_COPTS='--copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2
  --copt=-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8'
  echo "Building for the Pi Two/Three, with NEON acceleration"
fi

bazel build -c opt ${PI_COPTS} \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer --cpu=armeabi \
  --crosstool_top=@local_config_arm_compiler//:toolchain \
  --verbose_failures \
  //tensorflow/tools/benchmark:benchmark_model \
  //tensorflow/tools/pip_package:build_pip_package

OUTDIR=bazel-out/pi
mkdir -p ${OUTDIR}
echo "Final outputs will go to ${OUTDIR}"

# Build a universal wheel.
BDIST_OPTS="--universal" \
  bazel-bin/tensorflow/tools/pip_package/build_pip_package "${OUTDIR}"

OLD_FN=$(ls "${OUTDIR}" | grep \.whl)
SUB='s/tensorflow-([^-]+)-([^-]+)-.*/tensorflow-\1-\2-none-any.whl/; print'
NEW_FN=$(echo "${OLD_FN}" | perl -ne "${SUB}")
mv "${OUTDIR}/${OLD_FN}" "${OUTDIR}/${NEW_FN}"
cp bazel-bin/tensorflow/tools/benchmark/benchmark_model "${OUTDIR}"

echo "Output can be found here:"
find "${OUTDIR}"
