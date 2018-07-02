#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

function run_configure_for_cpu_build {
  yes "" | ./configure
}

function run_configure_for_gpu_build {
  # Enable CUDA support
  export TF_NEED_CUDA=1

  # TODO(pcloudy): Remove this after TensorFlow uses its own CRSOOTOOL
  # for GPU build on Windows
  export USE_MSVC_WRAPPER=1

  yes "" | ./configure
}

function set_remote_cache_options {
  echo "build --remote_instance_name=projects/tensorflow-testing-cpu" >> "${TMP_BAZELRC}"
  echo "build --experimental_remote_platform_override='properties:{name:\"build\" value:\"windows-x64\"}'" >> "${TMP_BAZELRC}"
  echo "build --remote_cache=remotebuildexecution.googleapis.com" >> "${TMP_BAZELRC}"
  echo "build --tls_enabled=true" >> "${TMP_BAZELRC}"
  echo "build --remote_timeout=3600" >> "${TMP_BAZELRC}"
  echo "build --auth_enabled=true" >> "${TMP_BAZELRC}"
  echo "build --spawn_strategy=remote" >> "${TMP_BAZELRC}"
  echo "build --strategy=Javac=remote" >> "${TMP_BAZELRC}"
  echo "build --strategy=Closure=remote" >> "${TMP_BAZELRC}"
  echo "build --genrule_strategy=remote" >> "${TMP_BAZELRC}"
  echo "build --google_credentials=$GOOGLE_CLOUD_CREDENTIAL" >> "${TMP_BAZELRC}"
}

function create_python_test_dir() {
  rm -rf "$1"
  mkdir -p "$1"
  cmd /c "mklink /J $1\\tensorflow .\\tensorflow"
}

function reinstall_tensorflow_pip() {
  echo "y" | pip uninstall tensorflow -q || true
  pip install ${1} --no-deps
}
