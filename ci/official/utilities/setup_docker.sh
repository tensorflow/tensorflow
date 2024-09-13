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
  # Simple retry logic for docker-pull errors. Sleeps if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$TFCI_DOCKER_IMAGE" || sleep 15
  docker pull "$TFCI_DOCKER_IMAGE" || sleep 30
  docker pull "$TFCI_DOCKER_IMAGE" || sleep 60
  docker pull "$TFCI_DOCKER_IMAGE"
fi 

echo "quoct building $TFCI_DOCKER_IMAGE"
DOCKER_BUILDKIT=1 docker build -t "$TFCI_DOCKER_IMAGE" $TFCI_DOCKER_REBUILD_ARGS

echo "quoct done building $TFCI_DOCKER_IMAGE"
docker image ls

# Keep the existing "tf" container if it's already present.
# The container is not cleaned up automatically! Remove it with:
# docker rm tf
if ! docker container inspect tf >/dev/null 2>&1 ; then
  # Pass all existing TFCI_ variables into the Docker container
  env_file=$(mktemp)
  env | grep ^TFCI_ > "$env_file"

  WORKING_DIR="$TFCI_GIT_DIR"
  if [[ `uname -s | grep -P '^MSYS_NT'` ]]; then
    env_file=$(cygpath -m $env_file)
    # Host dirs can only be mapped to an existing drive inside the container, so
    # T:\ is replaced with C:\.
    _TFCI_OUTPUT_DIR_WIN=$(replace_drive_letter_with_c "$TFCI_OUTPUT_DIR")
    sed -iE 's|^TFCI_OUTPUT_DIR=.*|TFCI_OUTPUT_DIR='"$_TFCI_OUTPUT_DIR_WIN"'|g' $env_file
    WORKING_DIR=$(replace_drive_letter_with_c "$TFCI_GIT_DIR")
    echo "GCE_METADATA_HOST=$IP_ADDR" > $env_file
  fi

  echo "quoct running bazel test"
  docker run $TFCI_DOCKER_ARGS --name tf -w "$WORKING_DIR" -itd --rm \
      -v "$TFCI_GIT_DIR:$WORKING_DIR" \
      --env-file "$env_file" \
      "$TFCI_DOCKER_IMAGE" \
    bash

  if [[ `uname -s | grep -P '^MSYS_NT'` ]]; then
    # Allow requests from the container.
    # Additional setup is contained in ci/official/envs/rbe.
    CONTAINER_IP_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' tf)
    netsh advfirewall firewall add rule name="Allow Metadata Proxy" dir=in action=allow protocol=TCP localport=80 remoteip="$CONTAINER_IP_ADDR"
  fi

  echo "quoct getting logs"
  docker logs -f tf

  echo "quoct hereeee"
  docker ps -a
  docker inspect tf

  docker exec --gpus all -v /home/kbuilder/.config/gcloud:/root/.config/gcloud --name tf -w /tmpfs/src/copybara -t --rm -v /tmpfs/src/copybara:/tmpfs/src/copybara --env-file "$env_file" gcr.io/tensorflow-sigs/build:674430781-python3.10 -- bazel test --repo_env=TF_PYTHON_VERSION=3.11 --config release_gpu_linux --config rbe_linux_cuda --profile /tmpfs/src/copybara/build_output/profile.json.gz --config=linux_cuda_pycpp_test

  sleep 10;
  echo "quoct inspecting"
  docker ps -a
  docker inspect tf

  sleep 60;
  echo "quoct inspecting"
  docker ps -a
  docker inspect tf

fi
tfrun() { docker exec tf "$@"; }
