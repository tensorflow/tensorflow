// RUN: tf-opt -tf-optimize %s -o %t && FileCheck %s < %t

// CHECK-LABEL: convbiasaddmul
func.func @convbiasaddmul(%arg: tensor<256x32x32x3xf32>) -> tensor<256x8x7x16xf32> {
  %filter = arith.constant dense<2.0> : tensor<3x3x3x16xf32>
  %bias = arith.constant dense<3.0> : tensor<16xf32>
  %value = arith.constant dense<4.0> : tensor<16xf32>
  %0 = "tf.Conv2D"(%arg, %filter) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
  %1 = "tf.BiasAdd"(%0, %bias) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"}: (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %2 = "tf.Mul"(%1, %value) {T = "tfdtype$DT_FLOAT"} : (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  func.return %2 : tensor<256x8x7x16xf32>

// CHECK-NEXT: %[[cst:.*]] = "tf.Const{{.*}} dense<8.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-NEXT: %[[cst_0:.*]] = "tf.Const{{.*}} dense<1.200000e+01> : tensor<16xf32>
// CHECK-NEXT: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK-NEXT: %[[bias:.*]] = "tf.AddV2"(%[[conv]], %[[cst_0]])
// CHECK-NEXT: return %[[bias]] : tensor<256x8x7x16xf32>
}

// CHECK-LABEL: convaddv2mul
func.func @convaddv2mul(%arg: tensor<256x32x32x3xf32>) -> tensor<256x8x7x16xf32> {
  %filter = arith.constant dense<2.0> : tensor<3x3x3x16xf32>
  %bias = arith.constant dense<3.0> : tensor<16xf32>
  %value = arith.constant dense<4.0> : tensor<16xf32>
  %0 = "tf.Conv2D"(%arg, %filter) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
  %1 = "tf.AddV2"(%0, %bias) {T = "tfdtype$DT_FLOAT"}: (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %2 = "tf.Mul"(%1, %value) {T = "tfdtype$DT_FLOAT"} : (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  func.return %2 : tensor<256x8x7x16xf32>

// CHECK-NEXT: %[[cst:.*]] = "tf.Const{{.*}} dense<8.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-NEXT: %[[cst_0:.*]] = "tf.Const{{.*}} dense<1.200000e+01> : tensor<16xf32>
// CHECK-NEXT: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK-NEXT: %[[add:.*]] = "tf.AddV2"(%[[conv]], %[[cst_0]])
// CHECK-NEXT: return %[[add]] : tensor<256x8x7x16xf32>
}

// CHECK-LABEL: fold_cast_fft_to_rfft
func.func @fold_cast_fft_to_rfft(%arg0: tensor<10x20x30xf32>) -> tensor<10x20x30xcomplex<f32>> {
  %0 = "tf.Cast"(%arg0) : (tensor<10x20x30xf32>) -> tensor<10x20x30xcomplex<f32>>
  %1 = "tf.FFT"(%0) : (tensor<10x20x30xcomplex<f32>>) -> tensor<10x20x30xcomplex<f32>>
  func.return %1: tensor<10x20x30xcomplex<f32>>

// CHECK:  %[[cst:.*]] = arith.constant dense<30> : tensor<1xi32>
// CHECK:  %[[rff:.*]] = "tf.RFFT"(%arg0, %[[cst]]) : (tensor<10x20x30xf32>, tensor<1xi32>) -> tensor<10x20x30xcomplex<f32>>
}

// CHECK-LABEL: not_fold_cast_fft_to_rfft
func.func @not_fold_cast_fft_to_rfft(%arg0: tensor<10x20x30xcomplex<f64>>) -> tensor<10x20x30xcomplex<f32>> {
  %0 = "tf.Cast"(%arg0) : (tensor<10x20x30xcomplex<f64>>) -> tensor<10x20x30xcomplex<f32>>
  %1 = "tf.FFT"(%0) : (tensor<10x20x30xcomplex<f32>>) -> tensor<10x20x30xcomplex<f32>>
  func.return %1: tensor<10x20x30xcomplex<f32>>

// CHECK: %[[fft:.*]] = "tf.FFT"(%0) : (tensor<10x20x30xcomplex<f32>>) -> tensor<10x20x30xcomplex<f32>>
}
