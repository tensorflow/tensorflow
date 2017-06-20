#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Script to upload release artifacts for the TensorFlow Java library to
# Maven Central. See README.md for an explanation.

TF_VERSION="$1"
SETTINGS_XML="$2"
shift
shift
CMD="$*"

if [[ -z "${TF_VERSION}" ]]
then
  echo "Usage: $0 <version to release> [<path to settings.xml>] ["bash" for debugging]"
  exit 1
fi

if [[ -z "${SETTINGS_XML}" ]]
then
  SETTINGS_XML="$HOME/.m2/settings.xml"
fi

if [[ -z "${CMD}" ]]
then
  CMD="bash run_inside_container.sh"
fi

if [[ ! -f "${SETTINGS_XML}" ]]
then
  echo "No settings.xml (containing credentials for upload) found"
  exit 1
fi

set -ex
docker run \
  -e TF_VERSION="${TF_VERSION}" \
  -v ${PWD}:/tensorflow \
  -v "${SETTINGS_XML}":/root/.m2/settings.xml \
  -v ${HOME}/.gnupg:/root/.gnupg \
  -w /tensorflow \
  -it \
  maven:3.3.9-jdk-8  \
  ${CMD}
