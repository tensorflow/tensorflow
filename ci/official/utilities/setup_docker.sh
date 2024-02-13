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
  # Simple retry logic for docker-pull errors. Sleeps for 15s if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$TFCI_DOCKER_IMAGE" || sleep 15
  docker pull "$TFCI_DOCKER_IMAGE" || sleep 15
  docker pull "$TFCI_DOCKER_IMAGE"
fi

if [[ "$TFCI_DOCKER_REBUILD_ENABLE" == 1 ]]; then
  DOCKER_BUILDKIT=1 docker build --cache-from "$TFCI_DOCKER_IMAGE" -t "$TFCI_DOCKER_IMAGE" $TFCI_DOCKER_REBUILD_ARGS
  if [[ "$TFCI_DOCKER_REBUILD_UPLOAD_ENABLE" == 1 ]]; then
    docker push "$TFCI_DOCKER_IMAGE"
  fi
fi

# Keep the existing "tf" container if it's already present.
# The container is not cleaned up automatically! Remove it with:
# docker rm tf
if ! docker container inspect tf >/dev/null 2>&1 ; then
  # Pass all existing TFCI_ variables into the Docker container
  env_file=$(mktemp)
  env | grep ^TFCI_ > "$env_file"
  docker run $TFCI_DOCKER_ARGS --name tf -w "$TFCI_GIT_DIR" -itd --rm \
      -v "$TFCI_GIT_DIR:$TFCI_GIT_DIR" \
      --env-file "$env_file" \
      "$TFCI_DOCKER_IMAGE" \
    bash
fi

tfrun() {
  # On Windows (where MSYSTEM will be set), always run under powershell if the
  # command is bazel-related, including bazel itself and any explicitly
  # requested bazel-bin/etc/etc targets. Bazel on Windows is supposed to be
  # invoked from powershell instead of MSYS environments; see
  # https://bazel.build/configure/windows
  # The fact that bazel-bin targets are included here somewhat contradicts
  # my comment in .bazelrc where I set --run_under to "exec" to force bazel
  # to use MSYS2. Maybe it's all to do with path conversion problems? I'm
  # not fully sure, but this combination did work when I was working on it.
  if [[ "$MSYSTEM" != "" && "$1" =~ bazel ]]; then
    docker exec tf powershell -command "$@"
  else
    docker exec tf "$@";
  fi
}
