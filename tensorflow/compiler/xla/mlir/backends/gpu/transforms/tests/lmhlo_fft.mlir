// RUN: xla-gpu-opt %s -xla-lmhlo-to-gpu-runtime | FileCheck %s

// CHECK: @compute(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<3x5x16x5xcomplex<f32>
// CHECK:   %[[ARG1:[a-z0-9]+]]: memref<3x5x16x8xf32>
// CHECK: )
func.func @compute(%arg0: memref<3x5x16x5xcomplex<f32>>,
                   %arg1: memref<3x5x16x8xf32>) {

  // CHECK: call @[[FFT:.*]](%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: fft_length = dense<[16, 8]> : tensor<2xi64>
  // CHECK-SAME: fft_type = #mhlo<fft_type IRFFT>
  "lmhlo.fft"(%arg0, %arg1) {
    fft_length = dense<[16, 8]> : tensor<2xi64>,
    fft_type = #mhlo<fft_type IRFFT>
  } : (memref<3x5x16x5xcomplex<f32>>, memref<3x5x16x8xf32>) -> ()

  // CHECK-NEXT: return
  func.return
}

// CHECK: func private @[[FFT]](memref<3x5x16x5xcomplex<f32>>,
// CHECK-SAME:                  memref<3x5x16x8xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.fft"}
