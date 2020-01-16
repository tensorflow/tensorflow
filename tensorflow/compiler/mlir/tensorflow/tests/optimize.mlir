// RUN: tf-opt -tf-optimize %s -o %t && FileCheck %s < %t

// CHECK-LABEL: convbiasaddmul
func @convbiasaddmul(%arg: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %filter = constant dense<2.0> : tensor<3x3x3x16xf32>
  %bias = constant dense<3.0> : tensor<16xf32>
  %value = constant dense<4.0> : tensor<16xf32>
  %0 = "tf.Conv2D"(%arg, %filter) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tf.BiasAdd"(%0, %bias) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"}: (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %2 = "tf.Mul"(%1, %value) {T = "tfdtype$DT_FLOAT"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %2 : tensor<256x30x30x16xf32>

// CHECK-NEXT: %[[cst:.*]] = "tf.Const{{.*}} dense<8.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-NEXT: %[[cst_0:.*]] = "tf.Const{{.*}} dense<1.200000e+01> : tensor<16xf32>
// CHECK-NEXT: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK-NEXT: %[[bias:.*]] = "tf.AddV2"(%[[conv]], %[[cst_0]])
// CHECK-NEXT: return %[[bias]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: convaddv2mul
func @convaddv2mul(%arg: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %filter = constant dense<2.0> : tensor<3x3x3x16xf32>
  %bias = constant dense<3.0> : tensor<16xf32>
  %value = constant dense<4.0> : tensor<16xf32>
  %0 = "tf.Conv2D"(%arg, %filter) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tf.AddV2"(%0, %bias) {T = "tfdtype$DT_FLOAT"}: (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %2 = "tf.Mul"(%1, %value) {T = "tfdtype$DT_FLOAT"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %2 : tensor<256x30x30x16xf32>

// CHECK-NEXT: %[[cst:.*]] = "tf.Const{{.*}} dense<8.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-NEXT: %[[cst_0:.*]] = "tf.Const{{.*}} dense<1.200000e+01> : tensor<16xf32>
// CHECK-NEXT: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[cst]])
// CHECK-NEXT: %[[add:.*]] = "tf.AddV2"(%[[conv]], %[[cst_0]])
// CHECK-NEXT: return %[[add]] : tensor<256x30x30x16xf32>
}
