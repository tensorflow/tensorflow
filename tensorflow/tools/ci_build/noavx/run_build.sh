#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
TF_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/tf_wheel}"
CACHE_DIR="${CACHE_DIR:-/tmp/tf_bazel_cache}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

echo "Building Docker image tf-cpu-noavx-builder..."
docker build -t tf-cpu-noavx-builder "${SCRIPT_DIR}"

echo "Running TensorFlow No-AVX build in container..."
docker run --rm \
  --name tf-cpu-noavx-build-run \
  -v "${TF_ROOT}:/tensorflow" \
  -v "${OUTPUT_DIR}:/tf_wheel" \
  -v "${CACHE_DIR}:/root/.cache" \
  tf-cpu-noavx-builder
