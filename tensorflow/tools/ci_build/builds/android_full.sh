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
configure_android_workspace

CPUS=armeabi-v7a,arm64-v8a,x86,x86_64

OUT_DIR="$(pwd)/out/"

rm -rf ${OUT_DIR}

# Build all relevant native libraries for each architecture.
for CPU in ${CPUS//,/ }
do
    echo "========== Building native libs for Android ${CPU} =========="
    bazel build -c opt --cpu=${CPU} \
        --crosstool_top=//external:android/crosstool \
        --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
        //tensorflow/core:android_tensorflow_lib \
        //tensorflow/contrib/android:libtensorflow_inference.so \
        //tensorflow/examples/android:libtensorflow_demo.so \
        //tensorflow/tools/benchmark:benchmark_model

    copy_lib bazel-bin/tensorflow/core/libandroid_tensorflow_lib.lo
    copy_lib bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so
    copy_lib bazel-bin/tensorflow/examples/android/libtensorflow_demo.so
    copy_lib bazel-bin/tensorflow/tools/benchmark/benchmark_model
done

# Build Jar and also demo containing native libs for all architectures.
# Enable sandboxing so that zip archives don't get incorrectly packaged
# in assets/ dir (see https://github.com/bazelbuild/bazel/issues/2334)
# TODO(gunan): remove extra flags once sandboxing is enabled for all builds.
echo "========== Building TensorFlow Android Jar and Demo =========="
bazel --bazelrc=/dev/null build -c opt --fat_apk_cpu=${CPUS} \
    --spawn_strategy=sandboxed --genrule_strategy=sandboxed \
    //tensorflow/contrib/android:android_tensorflow_inference_java \
    //tensorflow/examples/android:tensorflow_demo

echo "Copying demo and Jar to ${OUT_DIR}"
cp bazel-bin/tensorflow/examples/android/tensorflow_demo.apk \
    bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar ${OUT_DIR}

# Test Makefile build just to make sure it still works.
if [ -z "$NDK_ROOT" ]; then
   export NDK_ROOT=${ANDROID_NDK_HOME}
fi

echo "========== Benchmark Makefile Build Test =========="
tensorflow/contrib/makefile/build_all_android.sh

echo "========== Demo Makefile Build Test =========="
tensorflow/contrib/makefile/build_all_android.sh \
-s $(pwd)/tensorflow/contrib/makefile/sub_makefiles/android/Makefile.in \
-t "libtensorflow_inference.so libtensorflow_demo.so"

# Test Makefile build for tensorflow runtime with hexagon.
# -b ... build only, -p ... use prebuilt binaries
# This uses prebuilt binaries for hexagon dependencies because Building
# hexagon binaries from source code requires qualcomm sdk.
echo "========== Hexagon Build Test =========="
tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh -bp
