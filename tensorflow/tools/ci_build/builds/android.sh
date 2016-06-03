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

# Download model file.
# Note: This is workaround. This should be done by bazel.
model_file_name="inception5h.zip"
tmp_model_file_name="${HOME}/.cache/tensorflow_models/${model_file_name}"
mkdir -p $(dirname ${tmp_model_file_name})
[ -e "${tmp_model_file_name}" ] || wget -c "https://storage.googleapis.com/download.tensorflow.org/models/${model_file_name}" -O "${tmp_model_file_name}"
# We clean up after ourselves, but not if we exit with an error, so make sure we start clean
rm -rf tensorflow/examples/android/assets/
unzip -o "${tmp_model_file_name}" -d tensorflow/examples/android/assets/

# Modify the WORKSPACE file.
# Note: This is workaround. This should be done by bazel.
if grep -q '^android_sdk_repository' WORKSPACE && grep -q '^android_ndk_repository' WORKSPACE; then
  echo "You probably have your WORKSPACE file setup for Android."
else
  if [ -z "${ANDROID_API_LEVEL}" -o -z "${ANDROID_BUILD_TOOLS_VERSION}" ] || \
      [ -z "${ANDROID_SDK_HOME}" -o -z "${ANDROID_NDK_HOME}" ]; then
    echo "ERROR: Your WORKSPACE file does not seems to have proper android"
    echo "       configuration and not all the environment variables expected"
    echo "       inside ci_build android docker container are set."
    echo "       Please configure it manually. See: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/README.md"
  else
    cat << EOF >> WORKSPACE
android_sdk_repository(
    name = "androidsdk",
    api_level = ${ANDROID_API_LEVEL},
    build_tools_version = "${ANDROID_BUILD_TOOLS_VERSION}",
    path = "${ANDROID_SDK_HOME}",
)

android_ndk_repository(
    name="androidndk",
    path="${ANDROID_NDK_HOME}",
    api_level=21)
EOF
  fi
fi

# Build Android demo app.
bazel build -c opt --copt=-mfpu=neon //tensorflow/examples/android:tensorflow_demo

# Cleanup workarounds.
rm -rf tensorflow/examples/android/assets/
