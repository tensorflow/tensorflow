#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    --input_models=model1.tflite,model2.tflite \\"
  echo "    --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \\"
  echo "    --tflite_custom_ops_srcs=file1.cc,file2.h \\"
  echo "    --tflite_custom_ops_deps=dep1,dep2"
  echo ""
  echo "Where: "
  echo "  --input_models: Supported TFLite models. "
  echo "  --target_archs: Supported arches included in the aar file."
  echo "  --tflite_custom_ops_srcs: The src files for building additional TFLite custom ops if any."
  echo "  --tflite_custom_ops_deps: Dependencies for building additional TFLite custom ops if any."
  echo ""
  exit 1
}

function generate_list_field {
  local name="$1"
  local list_string="$2"
  local list=(${list_string//,/ })

  local message=("$name=[")
  for item in "${list[@]}"
  do
    message+=("\"$item\",")
  done
  message+=('],')
  printf '%s' "${message[@]}"
}

function print_output {
  if [ -z OMIT_PRINTING_OUTPUT_PATHS ]; then
    echo "Output can be found here:"
    for i in "$@"
    do
      # Check if the file exist.
      ls -1a ${ROOT_DIR}/$i
    done
  fi
}

function generate_tflite_aar {
  pushd ${TMP_DIR} > /dev/null
  # Generate the BUILD file.
  message=(
    'load("//tensorflow/lite:build_def.bzl", "tflite_custom_android_library")'
    'load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")'
    ''
    'tflite_custom_android_library('
    '    name = "custom_tensorflowlite",'
  )
  message+=('    '$(generate_list_field "models" $MODEL_NAMES))
  message+=('    '$(generate_list_field "srcs" $TFLITE_OPS_SRCS))
  message+=('    '$(generate_list_field "deps" $FLAG_TFLITE_OPS_DEPS))
  message+=(
    ')'
    ''
    'aar_with_jni('
    '    name = "tensorflow-lite",'
    '    android_library = ":custom_tensorflowlite",'
    ')'
    ''
  )
  printf '%s\n' "${message[@]}" >> BUILD

  # Build the aar package.
  popd > /dev/null
  # TODO(b/254278688): Enable 'xnn_enable_arm_fp16' with toolchain upgrade.
  bazel ${CACHE_DIR_FLAG} build -c opt --cxxopt='--std=c++17' \
        --fat_apk_cpu=${TARGET_ARCHS} \
        --define=xnn_enable_arm_fp16=false \
        --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
        //tmp:tensorflow-lite

   OUT_FILES="${OUT_FILES} bazel-bin/tmp/tensorflow-lite.aar"
}

function generate_flex_aar {
  pushd ${TMP_DIR}
  # Generating the BUILD file.
  message=(
    'load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_android_library")'
    'load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")'
    ''
    'tflite_flex_android_library('
    '    name = "custom_tensorflowlite_flex",'
    )
  message+=('    '$(generate_list_field "models" $MODEL_NAMES))
  message+=(
    ')'
    ''
    'aar_with_jni('
    '    name = "tensorflow-lite-select-tf-ops",'
    '    android_library = ":custom_tensorflowlite_flex",'
    ')'
  )
  printf '%s\n' "${message[@]}" >> BUILD

  cp ${ROOT_DIR}/tensorflow/lite/java/AndroidManifest.xml .
  cp ${ROOT_DIR}/tensorflow/lite/java/proguard.flags .
  popd

  # Build the aar package.
  # TODO(b/254278688): Enable 'xnn_enable_arm_fp16' with toolchain upgrade.
  bazel ${CACHE_DIR_FLAG} build -c opt --cxxopt='--std=c++17' \
      --fat_apk_cpu=${TARGET_ARCHS} \
      --define=xnn_enable_arm_fp16=false \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflow-lite-select-tf-ops

  OUT_FILES="${OUT_FILES} bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar"
}

# Check command line flags.
TARGET_ARCHS=x86,x86_64,arm64-v8a,armeabi-v7a
# If the environmant variable BAZEL_CACHE_DIR is set, use it as the user root
# directory of bazel.
if [ ! -z ${BAZEL_CACHE_DIR} ]; then
  CACHE_DIR_FLAG="--output_user_root=${BAZEL_CACHE_DIR}/cache"
fi

if [ "$#" -gt 4 ]; then
  echo "ERROR: Too many arguments."
  print_usage
fi

for i in "$@"
do
case $i in
    --input_models=*)
      FLAG_MODELS="${i#*=}"
      shift;;
    --target_archs=*)
      TARGET_ARCHS="${i#*=}"
      shift;;
    --tflite_custom_ops_srcs=*)
      FLAG_TFLITE_OPS_SRCS="${i#*=}"
      shift;;
    --tflite_custom_ops_deps=*)
      FLAG_TFLITE_OPS_DEPS="${i#*=}"
      shift;;
    *)
      echo "ERROR: Unrecognized argument: ${i}"
      print_usage;;
esac
done

# Check if users already run configure
cd $ROOT_DIR
if [ ! -f "$ROOT_DIR/.tf_configure.bazelrc" ]; then
  echo "ERROR: Please run ./configure first."
  exit 1
else
  if ! grep -q ANDROID_SDK_HOME "$ROOT_DIR/.tf_configure.bazelrc"; then
    echo "ERROR: Please run ./configure with Android config."
    exit 1
  fi
fi

# Build the standard aar package of no models provided.
# TODO(b/254278688): Enable 'xnn_enable_arm_fp16' with toolchain upgrade.
if [ -z ${FLAG_MODELS} ]; then
  bazel ${CACHE_DIR_FLAG} build -c opt --cxxopt='--std=c++17' \
    --config=monolithic \
    --fat_apk_cpu=${TARGET_ARCHS} \
    --define=xnn_enable_arm_fp16=false \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    //tensorflow/lite/java:tensorflow-lite

  print_output bazel-bin/tensorflow/lite/java/tensorflow-lite.aar
  exit 0
fi

# Prepare the tmp directory.
TMP_DIR="${ROOT_DIR}/tmp/"
rm -rf ${TMP_DIR} && mkdir -p ${TMP_DIR}

# Copy models to tmp directory.
MODEL_NAMES=""
for model in $(echo ${FLAG_MODELS} | sed "s/,/ /g")
do
  cp ${model} ${TMP_DIR}
  MODEL_NAMES="${MODEL_NAMES},$(basename ${model})"
done

# Copy srcs of additional tflite ops to tmp directory.
TFLITE_OPS_SRCS=""
for src_file in $(echo ${FLAG_TFLITE_OPS_SRCS} | sed "s/,/ /g")
do
  cp ${src_file} ${TMP_DIR}
  TFLITE_OPS_SRCS="${TFLITE_OPS_SRCS},$(basename ${src_file})"
done

# Build the custom aar package.
generate_tflite_aar

# Build flex aar if one of the models contain flex ops.
bazel ${CACHE_DIR_FLAG} build -c opt --config=monolithic \
  //tensorflow/lite/tools:list_flex_ops_no_kernel_main
bazel-bin/tensorflow/lite/tools/list_flex_ops_no_kernel_main \
  --graphs=${FLAG_MODELS} > ${TMP_DIR}/ops_list.txt
if [[ `cat ${TMP_DIR}/ops_list.txt` != "[]" ]]; then
  generate_flex_aar
fi

# List the output files.
rm -rf ${TMP_DIR}
print_output ${OUT_FILES}
