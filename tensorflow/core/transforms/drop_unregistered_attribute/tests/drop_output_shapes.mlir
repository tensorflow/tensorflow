// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tfg-transforms-opt %s --tfg-drop-unregistered-output-shapes=skip=tfg.placeholder | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: placeholder
  // CHECK: _output_shapes
  // CHECK: AddV2
  // CHECK-NOT: _output_shapes
  // CHECK: placeholder
  %arg0, %ctl = "tfg.placeholder"() {_output_shapes = ["tfshape$dim {size = 1}"] } : () -> (tensor<*xi32>, !tf_type.control)
  %add, %ctl3 = "tfg.AddV2"(%arg0, %arg1) {_output_shapes = ["tfshape$dim {size = 1}"] } : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, !tf_type.control)
  %arg1, %ctl2 = "tfg.placeholder"()  : () -> (tensor<*xi32>, !tf_type.control)
}

