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

set -e

copy_lib() {
  FILE=$1
  TARGET_DIR=${OUT_DIR}/native/$(basename $FILE)/${CPU}
  mkdir -p ${TARGET_DIR}
  echo "Copying ${FILE} to ${TARGET_DIR}"
  cp ${FILE} ${TARGET_DIR}
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"
# To setup Android via `configure` script.
export TF_SET_ANDROID_WORKSPACE=1
yes "" | ./configure

CPUS=armeabi-v7a,arm64-v8a

OUT_DIR="$(pwd)/out/"
AAR_LIB_TMP="$(pwd)/aar_libs"

rm -rf ${OUT_DIR}
rm -rf ${AAR_LIB_TMP}

# Build all relevant native libraries for each architecture.
for CPU in ${CPUS//,/ }
do
    echo "========== Building native libs for Android ${CPU} =========="
    bazel build --config=monolithic --cpu=${CPU} \
        --compilation_mode=opt --cxxopt=-std=c++17 \
        --distinct_host_configuration=true \
        --crosstool_top=//external:android/crosstool \
        --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
        //tensorflow/core:portable_tensorflow_lib \
        //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
        //tensorflow/tools/android/test:libtensorflow_demo.so \
        //tensorflow/tools/benchmark:benchmark_model

    copy_lib bazel-bin/tensorflow/tools/android/inference_interface/libtensorflow_inference.so
    copy_lib bazel-bin/tensorflow/tools/android/test/libtensorflow_demo.so
    copy_lib bazel-bin/tensorflow/tools/benchmark/benchmark_model

    mkdir -p ${AAR_LIB_TMP}/jni/${CPU}
    cp bazel-bin/tensorflow/tools/android/inference_interface/libtensorflow_inference.so ${AAR_LIB_TMP}/jni/${CPU}
done

# Build Jar and also demo containing native libs for all architectures.
# Enable sandboxing so that zip archives don't get incorrectly packaged
# in assets/ dir (see https://github.com/bazelbuild/bazel/issues/2334)
# TODO(gunan): remove extra flags once sandboxing is enabled for all builds.
echo "========== Building TensorFlow Android Jar and Demo =========="
bazel --bazelrc=/dev/null build --config=monolithic --fat_apk_cpu=${CPUS} \
    --compilation_mode=opt --cxxopt=-std=c++17 \
    --distinct_host_configuration=true \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --spawn_strategy=sandboxed --genrule_strategy=sandboxed \
    //tensorflow/tools/android/inference_interface:android_tensorflow_inference_java \
    //tensorflow/tools/android/inference_interface:android_tensorflow_inference_java.aar \
    //tensorflow/tools/android/test:tensorflow_demo

echo "Copying demo, AAR and Jar to ${OUT_DIR}"
cp bazel-bin/tensorflow/tools/android/test/tensorflow_demo.apk \
    bazel-bin/tensorflow/tools/android/inference_interface/libandroid_tensorflow_inference_java.jar ${OUT_DIR}

cp bazel-bin/tensorflow/tools/android/inference_interface/android_tensorflow_inference_java.aar \
   ${OUT_DIR}/tensorflow.aar

# TODO(andrewharp): build native libs into AAR directly once
# https://github.com/bazelbuild/bazel/issues/348 is resolved.
echo "Adding native libs to AAR"
chmod +w ${OUT_DIR}/tensorflow.aar
pushd ${AAR_LIB_TMP}
zip -ur ${OUT_DIR}/tensorflow.aar $(find jni -name *.so)
popd
rm -rf ${AAR_LIB_TMP}

# TODO(b/122377443): Restore Makefile builds after resolving r18b build issues.
