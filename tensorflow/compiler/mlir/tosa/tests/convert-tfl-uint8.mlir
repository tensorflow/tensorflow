// RUN: tf-opt --tosa-convert-tfl-uint8  --verify-each %s | FileCheck %s
// REQUIRES: tf_tosa

// Operations for testing --tosa-convert-tfl-uint8

// ----

// CHECK-LABEL: test_add_u8
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tfl.add
// CHECK: tosa.rescale
func.func @test_add_u8(%arg0: tensor<14x19x!quant.uniform<u8:f32, 0.015603500418365002:128>>, %arg1: tensor<14x19x!quant.uniform<u8:f32, 0.015612985007464886:127>>) -> tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>  {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<14x19x!quant.uniform<u8:f32, 0.015603500418365002:128>>, tensor<14x19x!quant.uniform<u8:f32, 0.015612985007464886:127>>) -> tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>
  func.return %0 : tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>
}
