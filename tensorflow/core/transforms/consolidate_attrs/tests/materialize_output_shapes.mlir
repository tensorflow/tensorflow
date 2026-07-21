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
// RUN: tfg-transforms-opt --tfg-prepare-attrs-export %s | FileCheck %s

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {_output_shapes = [#tf_type.shape<4>, #tf_type.shape<8>]}
  %A:2, %ctlA = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>, tensor<8xi32>)
  // CHECK: B {_output_shapes = [#tf_type.shape<*>, #tf_type.shape<2>]}
  %B:2, %ctlB = B {tfg.regenerate_output_shapes} : () -> (tensor<*xi32>, tensor<2xi32>)
  // Test excludes ops without regenerate attribute.
  // CHECK: C : ()
  %C:2, %ctlC = C : () -> (tensor<*xi32>, tensor<4xi32>)
}
