// RUN: tf-opt %s -tf-contraction-fusion | FileCheck %s

// CHECK-LABEL: matmulBiasAdd
func @matmulBiasAdd(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> tensor<8x64xf32> {
  // CHECK: %[[FUSED:.*]] = "tf._JitFusedMatMul"(%arg1, %arg2, %arg0)
  // CHECK-SAME: fusion = ["BiasAdd"]
  // CHECK-SAME: transpose_a = false, transpose_b = false
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<8x64xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<8x64xf32>, tensor<64xf32>) -> tensor<8x64xf32>
  // CHECK: return %[[FUSED]]
  return %4 : tensor<8x64xf32>
}

// CHECK-LABEL: matmulBiasAddRelu
func @matmulBiasAddRelu(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> tensor<8x64xf32> {
  // CHECK: %[[FUSED:.*]] = "tf._JitFusedMatMul"(%arg1, %arg2, %arg0)
  // CHECK-SAME: fusion = ["BiasAdd", "Relu"]
  // CHECK-SAME: transpose_a = false, transpose_b = false
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<8x64xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<8x64xf32>, tensor<64xf32>) -> tensor<8x64xf32>
  %5 = "tf.Relu"(%4) : (tensor<8x64xf32>) -> tensor<8x64xf32>
  // CHECK: return %[[FUSED]]
  return %5 : tensor<8x64xf32>
}

// CHECK-LABEL: matmulBiasAddLeakyRelu
func @matmulBiasAddLeakyRelu(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> tensor<8x64xf32> {
  // CHECK: %[[FUSED:.*]] = "tf._JitFusedMatMul"(%arg1, %arg2, %arg0)
  // CHECK-SAME: alpha = 2.000000e-01 : f32
  // CHECK-SAME: fusion = ["BiasAdd", "LeakyRelu"]
  // CHECK-SAME: transpose_a = false, transpose_b = false
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<8x64xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<8x64xf32>, tensor<64xf32>) -> tensor<8x64xf32>
  %5 = "tf.LeakyRelu"(%4) { alpha = 0.2 : f32 } : (tensor<8x64xf32>) -> tensor<8x64xf32>
  // CHECK: return %[[FUSED]]
  return %5 : tensor<8x64xf32>
}
