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

set -e

is_continuous_or_release() {
  [[ "$KOKORO_JOB_TYPE" == "CONTINUOUS_INTEGRATION" ]] || [[ "${KOKORO_JOB_TYPE}" == "RELEASE" ]]
}

# Move into the directory of the script
cd "$(dirname "$0")"

if is_continuous_or_release; then
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

# IMAGE="gcr.io/tensorflow-sigs/build-arm64:$TAG-$PYVER"
IMAGE="gcr.io/tensorflow-sigs/build-arm64:$TAG"
docker pull "$IMAGE" || true

gcloud auth configure-docker

# TODO(michaelhudgins): align with sig build and make it so not every python is
# being included in a single image
# --build-arg "PYTHON_VERSION=$PYVER" \
DOCKER_BUILDKIT=1 docker build \
  --cache-from "$IMAGE" \
  --target=devel \
  -t "$IMAGE"  .

docker push "$IMAGE"
