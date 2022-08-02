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

// RUN: lhlo-tfrt-opt %s -lmhlo-gpu-to-jitrt -split-input-file | FileCheck %s

// CHECK: @compute(
// CHECK:   %[[LHS:[a-z0-9]+]]: memref<4x4xf32>,
// CHECK:   %[[RHS:[a-z0-9]+]]: memref<4x4xf32>,
// CHECK:   %[[OUT:[a-z0-9]+]]: memref<4x4xf32>
// CHECK: )
func.func @compute(%lhs: memref<4x4xf32>, %rhs: memref<4x4xf32>,
                   %out: memref<4x4xf32>) {

  // CHECK: call @gemm(%[[LHS]], %[[RHS]], %[[OUT]])
  // CHECK-SAME:   algorithm = 13 : i64
  // CHECK-SAME:   alpha_imag = 0.000000e+00 : f64
  // CHECK-SAME:   alpha_real = 1.000000e+00 : f64
  // CHECK-SAME:   beta = 0.000000e+00 : f64
  // CHECK-SAME:   dot_dims = #mhlo.dot<lhs_contracting_dimensions = [1],
  // CHECK-SAME:                        rhs_contracting_dimensions = [0]>
  // CHECK-SAME:   uid = 0 : i64
  // CHECK-SAME: (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
  "lmhlo_gpu.gemm"(%lhs, %rhs, %out)
     {
       algorithm = 13 : i64,
       alpha_imag = 0.000000e+00 : f64,
       alpha_real = 1.000000e+00 : f64,
       batch_size = 1 : i64,
       beta = 0.000000e+00 : f64,
       dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                         rhs_contracting_dimensions = [0]>,
       lhs_stride = 16 : i64,
       rhs_stride = 16 : i64
     }
  : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK-NEXT: return
  func.return
}

// CHECK: func private @gemm(memref<4x4xf32>, memref<4x4xf32>,
// CHECK-SAME: memref<4x4xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.gemm"}
