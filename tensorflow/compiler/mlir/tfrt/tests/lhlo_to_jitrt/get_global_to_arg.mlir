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

// RUN: lhlo-tfrt-opt %s -lmhlo-constant-to-arg=min-num-elements=2 \
// RUN: | FileCheck %s

memref.global "private" constant @cst0 : memref<2x3xf32> =
  dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
         [4.000000e+00, 5.000000e+00, 6.000000e+00]]>

memref.global "private" constant @cst1 : memref<f32> =
  dense<1.000000e+00>

// CHECK:      @get_global(
// CHECK-SAME:   %[[ARG0:.*]]: memref<24xi8>
// CHECK-SAME:   %[[ARG1:.*]]: memref<4xi8>
// CHECK-SAME: ) -> (memref<2x3xf32>, memref<f32>) {
func.func @get_global(%arg0: memref<24xi8> {lmhlo.constant_name = "cst0"},
                      %arg1: memref<4xi8> {lmhlo.constant_name = "cst1"})
    -> (memref<2x3xf32>, memref<f32>) {

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = memref.view %[[ARG0]][%[[C0]]][] {{.*}} memref<2x3xf32>
  %0 = memref.get_global @cst0 : memref<2x3xf32>

  // CHECK: %[[V1:.*]] = memref.get_global {{.*}} : memref<f32>
  %1 = memref.get_global @cst1 : memref<f32>

  // CHECK: return %[[V0]], %[[V1]] : memref<2x3xf32>, memref<f32>
  return %0, %1 : memref<2x3xf32>, memref<f32>
}
