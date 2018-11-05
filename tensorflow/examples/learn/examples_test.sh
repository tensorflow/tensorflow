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

# This script exercises the examples of using TF.Learn.

DIR="$TEST_SRCDIR"

# Check if TEST_WORKSPACE is defined, and set as empty string if not.
if [ -z "${TEST_WORKSPACE-}" ]
then
  TEST_WORKSPACE=""
fi

if [ ! -z "$TEST_WORKSPACE" ]
then
  DIR="$DIR"/"$TEST_WORKSPACE"
fi

TFLEARN_EXAMPLE_BASE_DIR=$DIR/tensorflow/examples/learn


function test() {
  echo "Test $1:"
  $TFLEARN_EXAMPLE_BASE_DIR/$1 $2
  if [ $? -eq 0 ]
  then
    echo "Test passed."
    return 0
  else
    echo "Test failed."
    exit 1
  fi
}

test iris_custom_decay_dnn
test iris_custom_model
