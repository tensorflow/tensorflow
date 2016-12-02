#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Script to generate a tarball containing the TensorFlow C-library which
# consists of the C API header file and libtensorflow.so.
#
# Work in progress but this is a step towards a "binary" distribution of the
# TensorFlow C-library allowing TensorFlow language bindings to be used
# without having to recompile the TensorFlow framework from sources, which
# takes a while and also introduces many other dependencies.
#
# Usage:
# - Source this file in another bash script
# - Execute build_libtensorflow_tarball SUFFIX
#
# Produces: lib_package/libtensorflow${SUFFIX}.tar.gz
#
# ASSUMPTIONS:
# - build_libtensorflow_tarball is invoked from the root of the git tree.
# - Any environment variables needed by the "configure" script have been set.

function build_libtensorflow_tarball() {
  # Sanity check that this is being run from the root of the git repository.
  if [ ! -e "WORKSPACE" ]; then
    echo "Must run this from the root of the bazel workspace"
    exit 1
  fi
  TARBALL_SUFFIX="${1}"
  BAZEL="bazel --bazelrc ./tensorflow/tools/ci_build/install/.bazelrc"
  BAZEL_OPTS="-c opt"
  if [ "${TF_NEED_CUDA}" == "1" ]; then
    BAZEL_OPTS="${BAZEL_OPTS} --config=cuda"
  fi
  bazel clean --expunge
  yes "" | ./configure
  
  # TODO(ashankar): Once 
  # https://github.com/tensorflow/tensorflow/commit/1b32b698eddc10c0d85b0b8cf838f42023394de7  
  # can be undone, i.e., when bazel supports pkg_tar with python3+ then all of this below
  # can be replaced with something like:
  # bazel build ${BAZEL_OPTS} //tensorflow/tools/lib_package:libtensorflow.tar.gz
  
  bazel build ${BAZEL_OPTS} //tensorflow:libtensorflow.so
  DIR=lib_package
  rm -rf ${DIR}
  mkdir -p ${DIR}/build/lib
  mkdir -p ${DIR}/build/include/tensorflow/c
  cp bazel-bin/tensorflow/libtensorflow.so ${DIR}/build/lib
  cp tensorflow/c/c_api.h ${DIR}/build/include/tensorflow/c
  tar -C ${DIR}/build -cvf ${DIR}/libtensorflow${TARBALL_SUFFIX}.tar.gz include/tensorflow/c/c_api.h lib/libtensorflow.so
  rm -rf ${DIR}/build
}
