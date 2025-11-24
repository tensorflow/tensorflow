// RUN: tf-tosa-opt --tosa-convert-tfl-uint8  --verify-each %s | FileCheck %s


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

// ----

// CHECK-LABEL: test_cast_ui8
// CHECK-DAG: %[[multiplier:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-DAG: %[[shift:.+]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: %[[input_zp:.+]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: %[[output_zp:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: tosa.rescale %arg0, %[[multiplier]], %[[shift]], %[[input_zp]], %[[output_zp]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true}
// CHECK: tfl.cast
func.func @test_cast_ui8(%arg0: tensor<1x256x256x3x!quant.uniform<u8:f32, 0.015603500418365002:128>>) -> tensor<1x256x256x3xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<1x256x256x3x!quant.uniform<u8:f32, 0.015603500418365002:128>>) -> tensor<1x256x256x3xf32>
  func.return %0 : tensor<1x256x256x3xf32>
}
