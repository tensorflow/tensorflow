#!/bin/bash
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This script excercises the examples of using SkFlow.

source gbash.sh || exit
source module gbash_unit.sh

SKFLOW_EXAMPLE_BASE_DIR=$TEST_SRCDIR/tensorflow/examples/skflow

function test::examples::boston() {
  $SKFLOW_EXAMPLE_BASE_DIR/boston
}

function test::examples::iris() {
  $SKFLOW_EXAMPLE_BASE_DIR/iris
}

function test::examples::iris_custom_model() {
  $SKFLOW_EXAMPLE_BASE_DIR/iris_custom_model
}

gbash::unit::main "$@"
