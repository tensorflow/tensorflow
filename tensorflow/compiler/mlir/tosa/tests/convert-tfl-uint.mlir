// RUN: tf-tosa-opt --tosa-convert-tfl-uint-to-int --verify-diagnostics --verify-each %s | FileCheck %s


// Operations for testing --tosa-convert-tfl-uint-to-int

// -----

// CHECK-LABEL: test_add_u8
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tfl.add
// CHECK: tosa.rescale
func.func @test_add_u8(%arg0: tensor<14x19x!quant.uniform<u8:f32, 0.015603500418365002:128>>, %arg1: tensor<14x19x!quant.uniform<u8:f32, 0.015612985007464886:127>>) -> tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>  {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<14x19x!quant.uniform<u8:f32, 0.015603500418365002:128>>, tensor<14x19x!quant.uniform<u8:f32, 0.015612985007464886:127>>) -> tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>
  func.return %0 : tensor<14x19x!quant.uniform<u8:f32, 0.028094837442040443:127>>
}

// -----

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

// -----

// CHECK-LABEL: test_error_tosa_ops
func.func @test_error_tosa_ops(%arg0: tensor<5x10xi8>) -> (tensor<5x10xi8>, none) {

  // Dummy use to TFL dialect to load TFL dialect in MLIR context
  %0 = "tfl.no_value"() <{value}> : () -> none

  // expected-error @+1 {{tosa operations are not expected in this pass. Run tosa-convert-tfl-uint-to-int before tosa-legalize-tfl}}
  %cst1 = "tosa.const"() <{values = dense<1> : tensor<5x10xi8>}> : () -> tensor<5x10xi8>
  // expected-error @+1 {{tosa operations are not expected in this pass. Run tosa-convert-tfl-uint-to-int before tosa-legalize-tfl}}
  %1 = "tosa.add"(%arg0, %cst1) : (tensor<5x10xi8>, tensor<5x10xi8>) -> tensor<5x10xi8>


  func.return %1, %0 : tensor<5x10xi8>, none
}

// -----
// CHECK-LABEL: test_cast_ui32_with_zp
// expected-error @+1 {{Input argument has unsigned quantized type with zero point 128 which is not supported by TOSA for bitwidth 32.}}
func.func @test_cast_ui32_with_zp(%arg0: tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002:128>>) -> tensor<1x256x256x3xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002:128>>) -> tensor<1x256x256x3xf32>
  func.return %0 : tensor<1x256x256x3xf32>
}

// -----
// CHECK-LABEL:   func.func @test_cast_ui32(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002>>) -> tensor<1x256x256x3xf32> {
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[RESCALE_0:.*]] = tosa.rescale %[[ARG0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x256x256x3x!quant.uniform<i32:f32, 0.031207000843995952>>
// CHECK:           %[[VAL_4:.*]] = "tfl.cast"(%[[RESCALE_0]]) : (tensor<1x256x256x3x!quant.uniform<i32:f32, 0.031207000843995952>>) -> tensor<1x256x256x3xf32>
// CHECK:           return %[[VAL_4]]
func.func @test_cast_ui32(%arg0: tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002:0>>) -> tensor<1x256x256x3xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<1x256x256x3x!quant.uniform<u32:f32, 0.015603500418365002:0>>) -> tensor<1x256x256x3xf32>
  func.return %0 : tensor<1x256x256x3xf32>
}

// -----
// CHECK-LABEL:   func.func @test_cast_ui8_small_range(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x256x256x3x!quant.uniform<u8<10:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32> {
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-55> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[RESCALE_0:.*]] = tosa.rescale %[[ARG0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x256x256x3x!quant.uniform<u8<10:150>:f32, 0.015603500418365002:50>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x256x256x3x!quant.uniform<i8:f32, 0.0085666276806709817:-55>>
// CHECK:           %[[VAL_4:.*]] = "tfl.cast"(%[[RESCALE_0]]) : (tensor<1x256x256x3x!quant.uniform<i8:f32, 0.0085666276806709817:-55>>) -> tensor<1x256x256x3xf32>
// CHECK:           return %[[VAL_4]]
func.func @test_cast_ui8_small_range(%arg0: tensor<1x256x256x3x!quant.uniform<u8<10:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<1x256x256x3x!quant.uniform<u8<10:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32>
  func.return %0 : tensor<1x256x256x3xf32>
}

// -----
// CHECK-LABEL:   func.func @test_cast_ui8_narrow_range(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x256x256x3x!quant.uniform<u8<1:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32> {
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-43> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[RESCALE_0:.*]] = tosa.rescale %[[ARG0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = true} : (tensor<1x256x256x3x!quant.uniform<u8<1:150>:f32, 0.015603500418365002:50>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x256x256x3x!quant.uniform<i8<-127:127>:f32, 0.0091532344973873428:-43>>
// CHECK:           %[[VAL_4:.*]] = "tfl.cast"(%[[RESCALE_0]]) : (tensor<1x256x256x3x!quant.uniform<i8<-127:127>:f32, 0.0091532344973873428:-43>>) -> tensor<1x256x256x3xf32>
// CHECK:           return %[[VAL_4]]
func.func @test_cast_ui8_narrow_range(%arg0: tensor<1x256x256x3x!quant.uniform<u8<1:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<1x256x256x3x!quant.uniform<u8<1:150>:f32, 0.015603500418365002:50>>) -> tensor<1x256x256x3xf32>
  func.return %0 : tensor<1x256x256x3xf32>
}
