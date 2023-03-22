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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
TMP_DIR="tensorflow/lite/ios/tmp"
OUT_FILES=()

function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    --input_models=model1.tflite,model2.tflite \\"
  echo "    --target_archs=x86_64,armv7,arm64"
  echo ""
  echo "Where: "
  echo "  --input_models: Supported TFLite models."
  echo "  --target_archs: Supported arches included in the frameworks."
  echo "      Default: x86_64,armv7,arm64. i386 architecture is currently not"
  echo "      supported."
  echo ""
  exit 1
}

# generate_list_field takes two positional arguments:
# - Name of the field in the build rule.
# - Comma-separated list of values of this field.
# The function returns a string represents that field in the BUILD file. Ex:
# 'name = ["value1", "value2"],'
function generate_list_field {
  local name="$1"
  local list_string="$2"
  IFS=","
  read -ra list <<< "$list_string"

  local message=("$name=[")
  for item in "${list[@]}"
  do
    message+=("\"$item\",")
  done
  message+=('],')
  printf '%s' "${message[@]}"
}

# get_output_file_path takes one bazel target label as an argument, and prints
# the path of the first output file of the specified target.
function get_output_file_path {
  local starlark_file="${TMP_DIR}/print_output_file.starlark"
  cat > "${starlark_file}" << EOF
def format(target):
  return target.files.to_list()[0].path
EOF
  bazel cquery --config=ios $1 \
    --output=starlark --starlark:file="${starlark_file}" 2> /dev/null
}

function print_output {
  echo "Output can be found here:"
  for i in "${OUT_FILES[@]}"
  do
    # ls command returns failure if the file does not exist.
    ls -1a ${ROOT_DIR}/$i
  done
}

function generate_tflite_framework {
  pushd ${TMP_DIR} > /dev/null
  # Generate the BUILD file.
  message=(
    'load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")'
    'load("//tensorflow/lite:build_def.bzl", "tflite_custom_c_library")'
    'load("//tensorflow/lite/ios:ios.bzl", "TFL_MINIMUM_OS_VERSION")'
    'tflite_custom_c_library('
    '    name = "custom_c_api",'
    '    deps = ['
    '        "//tensorflow/lite/core/shims:builtin_ops_list",'
    '    ],'
    '    '"$(generate_list_field "models" "$MODEL_NAMES")"
    ')'
    'ios_static_framework('
    '    name = "TensorFlowLiteC_framework",'
    '    hdrs = ['
    '        "//tensorflow/lite/c:c_api_types.h",'
    '        "//tensorflow/lite/ios:common.h",'
    '        "//tensorflow/lite/ios:c_api.h",'
    '        "//tensorflow/lite/ios:xnnpack_delegate.h",'
    '    ],'
    '    bundle_name = "TensorFlowLiteC",'
    '    minimum_os_version = TFL_MINIMUM_OS_VERSION,'
    '    deps = ['
    '        ":custom_c_api",'
    '        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",'
    '    ],'
    ')'
  )
  printf '%s\n' "${message[@]}" >> BUILD

  # Build the framework package.
  local target="//${TMP_DIR}:TensorFlowLiteC_framework"
  popd > /dev/null
  bazel build -c opt --config=ios --ios_multi_cpus="${TARGET_ARCHS}" "${target}"

  OUT_FILES+=($(get_output_file_path "${target}"))
}

function generate_flex_framework {
  pushd ${TMP_DIR}
  # Generating the BUILD file.
  message=(
    'load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library")'
    'tflite_flex_cc_library('
    '   name = "custom_flex_delegate",'
    '    '"$(generate_list_field "models" "$MODEL_NAMES")"
    ')'
    'ios_static_framework('
    '    name = "TensorFlowLiteSelectTfOps_framework",'
    '    avoid_deps = ["//tensorflow/lite/c:common"],'
    '    bundle_name = "TensorFlowLiteSelectTfOps",'
    '    minimum_os_version = TFL_MINIMUM_OS_VERSION,'
    '    deps = ['
    '        ":custom_flex_delegate",'
    '    ],'
    ')'
  )
  printf '%s\n' "${message[@]}" >> BUILD
  popd

  # Build the framework.
  local target="//${TMP_DIR}:TensorFlowLiteSelectTfOps_framework"
  bazel build -c opt --config=ios --ios_multi_cpus="${TARGET_ARCHS}" "${target}"

  OUT_FILES+=($(get_output_file_path "${target}"))
}

# Check command line flags.
TARGET_ARCHS=x86_64,armv7,arm64

if [ "$#" -gt 2 ]; then
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
    *)
      echo "ERROR: Unrecognized argument: ${i}"
      print_usage;;
esac
done

cd $ROOT_DIR

# Check if users ran configure with iOS enabled.
if [ ! -f "$ROOT_DIR/TensorFlowLiteObjC.podspec" ]; then
  echo "ERROR: Please run ./configure with iOS config."
  exit 1
fi

# Prepare the tmp directory.
rm -rf ${TMP_DIR} && mkdir -p ${TMP_DIR}

# Copy models to tmp directory.
MODEL_NAMES=""
IFS=","
read -ra MODEL_PATHS <<< "${FLAG_MODELS}"
for model in "${MODEL_PATHS[@]}"
do
  cp ${model} ${TMP_DIR}
  if [ -z "$MODEL_NAMES" ]; then
    MODEL_NAMES="$(basename ${model})"
  else
    MODEL_NAMES="${MODEL_NAMES},$(basename ${model})"
  fi
done

# Build the custom framework.
generate_tflite_framework
if [ -z ${FLAG_MODELS} ]; then
  print_output
  exit 0
fi

# Build flex framework if one of the models contain flex ops.
bazel build -c opt --config=monolithic //tensorflow/lite/tools:list_flex_ops_no_kernel_main
bazel-bin/tensorflow/lite/tools/list_flex_ops_no_kernel_main --graphs=${FLAG_MODELS} > ${TMP_DIR}/ops_list.txt
if [[ `cat ${TMP_DIR}/ops_list.txt` != "[]" ]]; then
  generate_flex_framework
fi

rm -rf ${TMP_DIR}
print_output
