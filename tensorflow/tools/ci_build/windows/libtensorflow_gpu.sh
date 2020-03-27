#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

run_configure_for_gpu_build

# build_libtensorflow_tarball in ../builds/libtensorflow.sh
# cannot be used on Windows since it relies on pkg_tar rules.
# So we do something special here
bazel --output_user_root=${TMPDIR} build -c opt --copt=/arch:AVX --announce_rc --config=short_logs \
  :LICENSE \
  tensorflow:tensorflow.dll \
  tensorflow:tensorflow_dll_import_lib \
  tensorflow/tools/lib_package:clicenses_generate \
  tensorflow/java:tensorflow_jni.dll \
  tensorflow/tools/lib_package:jnilicenses_generate

DIR=lib_package
rm -rf ${DIR}
mkdir -p ${DIR}

# Zip up the .dll and the LICENSE for the JNI library.
cp bazel-bin/tensorflow/java/tensorflow_jni.dll ${DIR}/tensorflow_jni.dll
zip -j ${DIR}/libtensorflow_jni-gpu-windows-$(uname -m).zip \
  ${DIR}/tensorflow_jni.dll \
  bazel-bin/tensorflow/tools/lib_package/include/tensorflow/THIRD_PARTY_TF_JNI_LICENSES \
  LICENSE
rm -f ${DIR}/tensorflow_jni.dll

# Zip up the .dll, LICENSE and include files for the C library.
mkdir -p ${DIR}/include/tensorflow/c
mkdir -p ${DIR}/include/tensorflow/c/eager
mkdir -p ${DIR}/lib
cp bazel-bin/tensorflow/tensorflow.dll ${DIR}/lib/tensorflow.dll
cp bazel-bin/tensorflow/tensorflow.lib ${DIR}/lib/tensorflow.lib
cp tensorflow/c/c_api.h \
  tensorflow/c/tf_attrtype.h \
  tensorflow/c/tf_datatype.h \
  tensorflow/c/tf_status.h \
  tensorflow/c/tf_tensor.h \
  ${DIR}/include/tensorflow/c
cp tensorflow/c/eager/c_api.h ${DIR}/include/tensorflow/c/eager
cp LICENSE ${DIR}/LICENSE
cp bazel-bin/tensorflow/tools/lib_package/THIRD_PARTY_TF_C_LICENSES ${DIR}/
cd ${DIR}
zip libtensorflow-gpu-windows-$(uname -m).zip \
  lib/tensorflow.dll \
  lib/tensorflow.lib \
  include/tensorflow/c/eager/c_api.h \
  include/tensorflow/c/c_api.h \
  include/tensorflow/c/tf_attrtype.h \
  include/tensorflow/c/tf_datatype.h \
  include/tensorflow/c/tf_status.h \
  include/tensorflow/c/tf_tensor.h \
  LICENSE \
  THIRD_PARTY_TF_C_LICENSES
rm -rf lib include
