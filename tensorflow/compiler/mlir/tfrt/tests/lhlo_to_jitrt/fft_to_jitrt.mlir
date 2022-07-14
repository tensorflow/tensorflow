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
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<3x5x16x5xcomplex<f32>
// CHECK:   %[[ARG1:[a-z0-9]+]]: memref<3x5x16x8xf32>
// CHECK: )
func.func @compute(%arg0: memref<3x5x16x5xcomplex<f32>>,
                   %arg1: memref<3x5x16x8xf32>) {

  // CHECK: call @[[FFT:.*]](%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: fft_length = dense<[16, 8]> : tensor<2xi64>
  // CHECK-SAME: fft_type = 3 : i32
  "lmhlo.fft"(%arg0, %arg1) {
    fft_length = dense<[16, 8]> : tensor<2xi64>,
    fft_type = #mhlo<fft_type IRFFT>
  } : (memref<3x5x16x5xcomplex<f32>>, memref<3x5x16x8xf32>) -> ()

  // CHECK-NEXT: return
  func.return
}

// CHECK: func private @[[FFT]](memref<3x5x16x5xcomplex<f32>>,
// CHECK-SAME:                  memref<3x5x16x8xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.fft"}
