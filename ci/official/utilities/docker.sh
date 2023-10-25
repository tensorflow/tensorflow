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
if [[ "$TFCI_DOCKER_PULL_ENABLE" == 1 ]]; then
  docker pull "$TFCI_DOCKER_IMAGE"
fi

if [[ "$TFCI_DOCKER_REBUILD_ENABLE" == 1 ]]; then
  DOCKER_BUILDKIT=1 docker build --cache-from "$TFCI_DOCKER_IMAGE" -t "$TFCI_DOCKER_IMAGE" "${TFCI_DOCKER_REBUILD_ARGS[@]}"
  if [[ "$TFCI_DOCKER_REBUILD_UPLOAD_ENABLE" == 1 ]]; then
    docker push "$TFCI_DOCKER_IMAGE"
  fi
fi

# Keep the existing "tf" container if it's already present.
# The container is not cleaned up automatically! Remove it with:
# docker rm tf
if ! docker container inspect tf >/dev/null 2>&1 ; then
  docker run "${TFCI_DOCKER_ARGS[@]}" --name tf -w "$TFCI_GIT_DIR" -itd --rm \
      -v "$TFCI_GIT_DIR:$TFCI_GIT_DIR" \
      --env TFCI_PYTHON_VERSION \
      "$TFCI_DOCKER_IMAGE" \
    bash
fi
tfrun() { docker exec tf "$@"; }
