// RUN: xla-cpu-opt %s -xla-legalize-collective-ops | FileCheck %s

func.func @fft(%arg0: tensor<3x5x4x8x256xf32>) -> tensor<3x5x4x8x129xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) {
    fft_length = dense<[4, 8, 256]> : tensor<3xi64>,
    fft_type = #mhlo<fft_type RFFT>
  } : (tensor<3x5x4x8x256xf32>) -> tensor<3x5x4x8x129xcomplex<f32>>
  func.return %0 : tensor<3x5x4x8x129xcomplex<f32>>
}

// CHECK-LABEL: @fft
//  CHECK-SAME: %[[ARG0:.*]]: tensor
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<3x5x4x8x129xcomplex<f32>>
//       CHECK: %[[FFT:.*]] = "xla_cpu.fft"(%[[ARG0]], %[[DST]])
//  CHECK-SAME: {fft_length = [4, 8, 256], fft_type = 2 : i32}
//       CHECK: return %[[FFT]]
