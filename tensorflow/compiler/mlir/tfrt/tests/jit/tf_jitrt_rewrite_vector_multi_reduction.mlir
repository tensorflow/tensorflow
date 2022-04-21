// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-tfrt-opt %s --tf-jitrt-rewrite-vector-multi-reduction \
// RUN: | FileCheck %s

// CHECK-LABEL: func @vector_row
func.func @vector_row(%arg0: vector<2x4xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0 [1] : vector<2x4xf32> to vector<2xf32>
    func.return %0 : vector<2xf32>
}
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.mulf

// CHECK-LABEL: func @vector_col
func.func @vector_col(%arg0: vector<2x4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0 [0] : vector<2x4xf32> to vector<4xf32>
    func.return %0 : vector<4xf32>
}
// CHECK: arith.mulf

// CHECK-LABEL: func @vector_1d
func.func @vector_1d(%arg0: vector<4xf32>) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0 [0] : vector<4xf32> to f32
    func.return %0 : f32
}
// CHECK: vector.reduction <mul>
