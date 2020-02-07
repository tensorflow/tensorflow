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
# Script for helping to record method for building the RBE docker images.
#
# The first argument to the script is expected to be the name of the docker file
# to build. Example:
#
# $ ./build_rbe.sh Dockerfile.rbe.ubuntu16.04-manylinux2010

function main() {
  set -eu

  cd "${0%/*}"

  local DOCKERFILE="$(basename "$1")"
  if [[ ! -e "$DOCKERFILE" ]]; then
    echo "$DOCKERFILE does not exist in $PWD" >> /dev/stderr
    exit 1
  fi

  local IMAGE_NAME_SUFFIX="${1#Dockerfile.rbe.}"
  if [[ "$IMAGE_NAME_SUFFIX" == "$DOCKERFILE" ]]; then
    echo 'File must start with "Dockerfile.rbe."' >> /dev/stderr
    exit 1
  fi

  local ARGS=(
    --config=cloudbuild.yaml
    --machine-type=n1-highcpu-32
    --substitutions=_DOCKERFILE="$1",_IMAGE_NAME="nosla-$IMAGE_NAME_SUFFIX"
    --timeout=1h
  )

  gcloud --project=tensorflow-testing builds submit "${ARGS[@]}" .
}

main "$@"
