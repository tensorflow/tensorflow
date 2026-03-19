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

if [ ! -e "WORKSPACE" ]; then
  echo "Must run this from the root of the bazel workspace"
  echo "Currently at ${PWD}"
  exit 1
fi

# build_libtensorflow_tarball in ../builds/libtensorflow.sh
# cannot be used on Windows since it relies on pkg_tar rules.
# So we do something special here
bazel --output_user_root=${TMPDIR} build \
  -c opt \
  --copt=/arch:AVX \
  --announce_rc \
  --config=short_logs \
  --config=win_clang \
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
zip -j ${DIR}/libtensorflow_jni-cpu-windows-$(uname -m).zip \
  ${DIR}/tensorflow_jni.dll \
  bazel-bin/tensorflow/tools/lib_package/include/tensorflow/THIRD_PARTY_TF_JNI_LICENSES \
  LICENSE
rm -f ${DIR}/tensorflow_jni.dll

# Zip up the .dll, LICENSE and include files for the C library.
mkdir -p ${DIR}/include/tensorflow/c
mkdir -p ${DIR}/include/tensorflow/c/eager
mkdir -p ${DIR}/include/tensorflow/core/platform
mkdir -p ${DIR}/include/xla/tsl/c
mkdir -p ${DIR}/include/tsl/platform
mkdir -p ${DIR}/lib
cp bazel-bin/tensorflow/tensorflow.dll ${DIR}/lib/tensorflow.dll
cp bazel-bin/tensorflow/tensorflow.lib ${DIR}/lib/tensorflow.lib
cp tensorflow/c/c_api.h \
  tensorflow/c/tf_attrtype.h \
  tensorflow/c/tf_buffer.h  \
  tensorflow/c/tf_datatype.h \
  tensorflow/c/tf_status.h \
  tensorflow/c/tf_tensor.h \
  tensorflow/c/tf_tensor_helper.h \
  tensorflow/c/tf_tstring.h \
  tensorflow/c/tf_file_statistics.h \
  tensorflow/c/tensor_interface.h \
  tensorflow/c/c_api_macros.h \
  tensorflow/c/c_api_experimental.h \
  ${DIR}/include/tensorflow/c
cp tensorflow/c/eager/c_api.h \
  tensorflow/c/eager/c_api_experimental.h \
  tensorflow/c/eager/dlpack.h \
  ${DIR}/include/tensorflow/c/eager
cp tensorflow/core/platform/ctstring.h \
  tensorflow/core/platform/ctstring_internal.h \
  ${DIR}/include/tensorflow/core/platform
cp third_party/xla/xla/tsl/c/tsl_status.h ${DIR}/include/xla/tsl/c
cp third_party/xla/third_party/tsl/tsl/platform/ctstring.h \
   third_party/xla/third_party/tsl/tsl/platform/ctstring_internal.h \
   ${DIR}/include/tsl/platform
cp LICENSE ${DIR}/LICENSE
cp bazel-bin/tensorflow/tools/lib_package/THIRD_PARTY_TF_C_LICENSES ${DIR}/
cd ${DIR}
zip libtensorflow-cpu-windows-$(uname -m).zip \
  lib/tensorflow.dll \
  lib/tensorflow.lib \
  include/tensorflow/c/eager/c_api.h \
  include/tensorflow/c/eager/c_api_experimental.h \
  include/tensorflow/c/eager/dlpack.h \
  include/tensorflow/c/c_api.h \
  include/tensorflow/c/tf_attrtype.h \
  include/tensorflow/c/tf_buffer.h  \
  include/tensorflow/c/tf_datatype.h \
  include/tensorflow/c/tf_status.h \
  include/tensorflow/c/tf_tensor.h \
  include/tensorflow/c/tf_tensor_helper.h \
  include/tensorflow/c/tf_tstring.h \
  include/tensorflow/c/tf_file_statistics.h \
  include/tensorflow/c/tensor_interface.h \
  include/tensorflow/c/c_api_macros.h \
  include/tensorflow/c/c_api_experimental.h \
  include/tensorflow/core/platform/ctstring.h \
  include/tensorflow/core/platform/ctstring_internal.h \
  include/xla/tsl/c/tsl_status.h \
  include/tsl/platform/ctstring.h \
  include/tsl/platform/ctstring_internal.h \
  LICENSE \
  THIRD_PARTY_TF_C_LICENSES
rm -rf lib include

cd ..
tar -zcvf windows_cpu_libtensorflow_binaries.tar.gz lib_package
