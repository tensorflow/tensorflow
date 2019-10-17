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
# Creates the project file distributions for the TensorFlow Lite Micro test and
# example targets aimed at embedded platforms.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../../..
cd ${ROOT_DIR}
pwd

python -m pip install six

make -f tensorflow/lite/experimental/micro/tools/make/Makefile \
  clean clean_downloads

make -f tensorflow/lite/experimental/micro/tools/make/Makefile \
  TARGET="arduino" \
  TAGS="portable_optimized" \
  generate_arduino_zip

tensorflow/lite/experimental/micro/tools/ci_build/install_arduino_cli.sh

tensorflow/lite/experimental/micro/tools/ci_build/test_arduino_library.sh \
  tensorflow/lite/experimental/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip

# Needed to solve CI build bug triggered by files added to source tree.
make -f tensorflow/lite/experimental/micro/tools/make/Makefile clean_downloads
