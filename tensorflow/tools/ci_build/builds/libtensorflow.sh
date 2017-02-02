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

  # Remove this test call when
  # https://github.com/bazelbuild/bazel/issues/2352
  # and https://github.com/bazelbuild/bazel/issues/1580
  # have been resolved and the "manual" tags on the BUILD targets
  # in tensorflow/tools/lib_package/BUILD are removed.
  # Till then, must manually run the test.
  bazel test ${BAZEL_OPTS} //tensorflow/tools/lib_package:libtensorflow_test

  bazel build ${BAZEL_OPTS} //tensorflow/tools/lib_package:libtensorflow.tar.gz
  DIR=lib_package
  mkdir -p ${DIR}
  cp bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz ${DIR}/libtensorflow${TARBALL_SUFFIX}.tar.gz
}
