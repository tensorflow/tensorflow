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

// ----

// CHECK-LABEL:   func.func @test_tosa_const_ui8() -> (tensor<!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<2x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<!quant.uniform<i8:f32, 0.028094837442040443:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.028094837442040443>>) {
// CHECK:           %[[CST_0D_UI8:.*]] = "tosa.const"() <{values = dense<-127> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK:           %[[CST_2D_UI8:.*]] = "tosa.const"() <{values = dense<{{\[\[}}-128, 0, 127], [-118, -108, -98]]> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK:           %[[CST_0D_QUANT_U8:.*]] = "tosa.const"() <{values = dense<-28> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 0.028094837442040443:-1>>
// CHECK:           %[[CST_2D_QUANT_U8:.*]] = "tosa.const"() <{values = dense<{{\[\[}}-128, -1], [72, 127]]> : tensor<2x2xi8>}> : () -> tensor<2x2x!quant.uniform<i8:f32, 0.028094837442040443>>
// CHECK:           return %[[CST_0D_UI8]], %[[CST_2D_UI8]], %[[CST_0D_QUANT_U8]], %[[CST_2D_QUANT_U8]] : tensor<!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<2x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<!quant.uniform<i8:f32, 0.028094837442040443:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.028094837442040443>>
// CHECK:         }
func.func @test_tosa_const_ui8() -> (
    tensor<ui8>,
    tensor<2x3xui8>,
    tensor<!quant.uniform<u8:f32, 0.028094837442040443:127>>,
    tensor<2x2x!quant.uniform<u8:f32, 0.028094837442040443:128>>
) {

  %0 = "tosa.const"() {
    values = dense<1> : tensor<ui8>
  } : () -> tensor<ui8>
  

  %1 = "tosa.const"() {
    values = dense<[[0, 128, 255], [10, 20, 30]]> : tensor<2x3xui8>
  } : () -> tensor<2x3xui8>


  %2 = "tosa.const"() {
    values = dense<100> : tensor<ui8>
  } : () -> tensor<!quant.uniform<u8:f32, 0.028094837442040443:127>>


  %3 = "tosa.const"() {
    values = dense<[[0, 127], [200, 255]]> : tensor<2x2xui8>
  } : () -> tensor<2x2x!quant.uniform<u8:f32, 0.028094837442040443:128>>

  return %0, %1, %2, %3 : 
    tensor<ui8>,
    tensor<2x3xui8>,
    tensor<!quant.uniform<u8:f32, 0.028094837442040443:127>>,
    tensor<2x2x!quant.uniform<u8:f32, 0.028094837442040443:128>>
}
