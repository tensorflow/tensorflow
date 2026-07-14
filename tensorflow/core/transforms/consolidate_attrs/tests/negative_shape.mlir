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
// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {tfg.regenerate_output_shapes}
  // CHECK-SAME: tensor<?xi32>, tensor<4xi32>, tensor<?xi32>
  %A:3, %ctl = A {
    _output_shapes = [#tf_type.shape<-4>, #tf_type.shape<4>, #tf_type.shape<?>]
  } : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
}
