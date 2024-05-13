#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# Builds the following Docker images for Linux ARM64. See the accompanying
# Dockerfile for more details:
# - gcr.io/tensorflow-sigs/build-arm64:jax-latest-multi-python
# - gcr.io/tensorflow-sigs/build-arm64:tf-latest-multi-python

set -exo pipefail

function is_continuous_or_release() {
  [[ "$KOKORO_JOB_TYPE" == "CONTINUOUS_INTEGRATION" ]] || [[ "$KOKORO_JOB_TYPE" == "RELEASE" ]]
}

# Move into the directory of the script
cd "$(dirname "$0")"

if is_continuous_or_release || [[ -z "$KOKORO_BUILD_ID" ]]; then
  # A continuous job is the only one to publish to latest
  TAG="latest-multi-python"
else
  # If it is a change, grab a good tag for iterative builds
  if [[ -z "${KOKORO_GITHUB_PULL_REQUEST_NUMBER}" ]]; then
    TAG=$(head -n 1 "$KOKORO_PIPER_DIR/presubmit_request.txt" | cut -d" " -f2)
  else
    TAG="pr-${KOKORO_GITHUB_PULL_REQUEST_NUMBER}"
  fi
fi

# Build for both JAX and TF usage.  We do these in one place because they share
# almost all of the same cache layers
export DOCKER_BUILDKIT=1
for target in jax tf; do
  IMAGE="gcr.io/tensorflow-sigs/build-arm64:$target-$TAG"
  docker pull "$IMAGE" || true
  # Due to some flakiness of resources pulled in the build, allow the docker
  # command to reattempt build a few times in the case of failure (b/302558736)
  set +e
  for i in $(seq 1 5)
  do
    docker build \
    --build-arg REQUIREMENTS_FILE=jax.requirements.txt \
    --target=$target \
    --cache-from "$IMAGE" \
    -t "$IMAGE"  . && break
  done
  final=$?
  if [ $final -ne 0 ]; then
    exit $final
  fi
  set -e

  if [[ -n "$KOKORO_BUILD_ID" ]]; then
    gcloud auth configure-docker
    docker push "$IMAGE"
  fi
done
