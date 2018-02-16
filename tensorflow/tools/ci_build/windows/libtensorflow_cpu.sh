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
#
# Script to produce binary release of libtensorflow (C API, Java jars etc.).

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup environment for bazel builds
source "${SCRIPT_DIR}/bazel/common_env.sh"
source "${SCRIPT_DIR}/bazel/bazel_test_lib.sh"

# Sanity check that this is being run from the root of the git repository.
cd ${SCRIPT_DIR}/../../../..
if [ ! -e "WORKSPACE" ]; then
  echo "Must run this from the root of the bazel workspace"
  echo "Currently at ${PWD}, script is at ${SCRIPT_DIR}"
  exit 1
fi

export TF_BAZEL_TARGETS="//tensorflow:libtensorflow.so"
export TF_BAZEL_TARGETS="${TF_BAZEL_TARGETS} //tensorflow/tools/lib_package:clicenses_generate"
export TF_BAZEL_TARGETS="${TF_BAZEL_TARGETS} //tensorflow/java:libtensorflow_jni.so"
export TF_BAZEL_TARGETS="${TF_BAZEL_TARGETS} //tensorflow/tools/lib_package:jnilicenses_generate"

run_configure_for_cpu_build

# build_libtensorflow_tarball in ../builds/libtensorflow.sh
# cannot be used on Windows since it relies on pkg_tar rules.
# So we do something special here
bazel build -c opt --copt=/arch:AVX \
  tensorflow:libtensorflow.so \
  tensorflow/tools/lib_package:clicenses_generate \
  tensorflow/java:libtensorflow_jni.so \
  tensorflow/tools/lib_package:jnilicenses_generate

DIR=lib_package
rm -rf ${DIR}
mkdir -p ${DIR}

# Zip up the .dll and the LICENSE for the JNI library.
cp bazel-bin/tensorflow/java/libtensorflow_jni.so ${DIR}/tensorflow_jni.dll
zip -j ${DIR}/libtensorflow_jni-cpu-windows-$(uname -m).zip \
  ${DIR}/tensorflow_jni.dll \
  bazel-genfiles/tensorflow/tools/lib_package/include/tensorflow/jni/LICENSE
rm -f ${DIR}/tensorflow_jni.dll

# Zip up the .dll, LICENSE and include files for the C library.
mkdir -p ${DIR}/include/tensorflow/c
mkdir -p ${DIR}/include/tensorflow/c/eager
mkdir -p ${DIR}/lib
cp bazel-bin/tensorflow/libtensorflow.so ${DIR}/lib/tensorflow.dll
cp tensorflow/c/c_api.h ${DIR}/include/tensorflow/c
cp tensorflow/c/eager/c_api.h ${DIR}/include/tensorflow/c/eager
cp bazel-genfiles/tensorflow/tools/lib_package/include/tensorflow/c/LICENSE ${DIR}/include/tensorflow/c
cd ${DIR}
zip libtensorflow-cpu-windows-$(uname -m).zip \
  lib/tensorflow.dll \
  include/tensorflow/c/eager/c_api.h \
  include/tensorflow/c/c_api.h \
  include/tensorflow/c/LICENSE
rm -rf lib include
