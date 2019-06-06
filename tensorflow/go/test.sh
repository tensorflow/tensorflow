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

# TensorFlow uses 'bazel' for builds and tests.
# The TensorFlow Go API aims to be usable with the 'go' tool
# (using 'go get' etc.) and thus without bazel.
#
# This script acts as a brige between bazel and go so that:
#   bazel test :test
# succeeds iff
#   go test github.com/tensorflow/tensorflow/tensorflow/go
# succeeds.

set -ex

# Find the 'go' tool
if [[ ! -x "go" && -z $(which go) ]]
then
  if [[ -x "/usr/local/go/bin/go" ]]
  then
    export PATH="${PATH}:/usr/local/go/bin"
  else
    echo "Could not find the 'go' tool in PATH or /usr/local/go"
    exit 1
  fi
fi

# Setup a GOPATH that includes just the TensorFlow Go API.
export GOPATH="${TEST_TMPDIR}/go"
export GOCACHE="${TEST_TMPDIR}/cache"
mkdir -p "${GOPATH}/src/github.com/tensorflow"
ln -s -f "${PWD}" "${GOPATH}/src/github.com/tensorflow/tensorflow"

# Ensure that the TensorFlow C library is accessible to the
# linker at build and run time.
export LIBRARY_PATH="${PWD}/tensorflow"
OS=$(uname -s)
if [[ "${OS}" = "Linux" ]]
then
  if [[ -z "${LD_LIBRARY_PATH}" ]]
  then
    export LD_LIBRARY_PATH="${PWD}/tensorflow"
  else
    export LD_LIBRARY_PATH="${PWD}/tensorflow:${LD_LIBRARY_PATH}"
  fi
elif [[ "${OS}" = "Darwin" ]]
then
  if [[ -z "${DYLD_LIBRARY_PATH}" ]]
  then
    export DYLD_LIBRARY_PATH="${PWD}/tensorflow"
  else
    export DYLD_LIBRARY_PATH="${PWD}/tensorflow:${DYLD_LIBRARY_PATH}"
  fi
else 
  echo "Only support Linux/Darwin, System $OS is not supported"
  exit 1
fi

# Document the Go version and run tests
echo "Go version: $(go version)"
go test \
  github.com/tensorflow/tensorflow/tensorflow/go  \
  github.com/tensorflow/tensorflow/tensorflow/go/op
