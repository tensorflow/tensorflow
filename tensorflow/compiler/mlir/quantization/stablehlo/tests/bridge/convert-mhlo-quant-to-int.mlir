// RUN: stablehlo-quant-opt "-convert-mhlo-quant-to-int=legalize-chlo=false" -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @uniform_quantize_and_dequantize
func.func @uniform_quantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[HALF]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.floor %[[VAL1]] : tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2]] : (tensor<?x?xf32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_maximum %[[VAL4]], %[[QUANT_MIN]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_minimum %[[VAL5]], %[[QUANT_MAX]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL8]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL11:.*]] = chlo.broadcast_multiply %[[VAL10]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL11]] : tensor<?x?xf32>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_convert_dequantize
func.func @uniform_quantize_convert_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[HALF]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.floor %[[VAL1]] : tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2]] : (tensor<?x?xf32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_maximum %[[VAL4]], %[[QUANT_MIN]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_minimum %[[VAL5]], %[[QUANT_MAX]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : tensor<?x?xi8>
  %1 = mhlo.convert %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xi8>

  // CHECK: %[[VAL9:.*]] = mhlo.convert %[[VAL8]] : tensor<?x?xi8>
  %2 = mhlo.convert %1 : (tensor<?x?xi8>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = chlo.broadcast_subtract %[[VAL10]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL12:.*]] = mhlo.convert %[[VAL11]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL13:.*]] = chlo.broadcast_multiply %[[VAL12]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL13]] : tensor<?x?xf32>
  %3 = mhlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_int4
func.func @uniform_quantize_and_dequantize_int4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-8> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[HALF]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.floor %[[VAL1]] : tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2]] : (tensor<?x?xf32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_maximum %[[VAL4]], %[[QUANT_MIN]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_minimum %[[VAL5]], %[[QUANT_MAX]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL8]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL11:.*]] = chlo.broadcast_multiply %[[VAL10]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL11]] : tensor<?x?xf32>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_type_exensions
func.func @uniform_quantize_and_dequantize_type_exensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xi8, #mhlo.type_extensions<bounds = [4, 4]>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>, tensor<f32>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?xi32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?xi8, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>, tensor<f32>) -> tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add
func.func @uniform_quantize_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add_i32
func.func @uniform_quantize_add_i32(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1:.*]], %[[VAL3:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-NEXT: return
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>,tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add_int4
func.func @uniform_quantize_add_int4(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: @uniform_quantize_add_different_lhs_type
func.func @uniform_quantize_add_different_lhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[INPUT_ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[INPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[MULTIPLIER:.*]] = mhlo.constant dense<16384> : tensor<i32>
  // CHECK-DAG: %[[TOTAL_SHIFT:.*]] = mhlo.constant dense<13> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<4096> : tensor<i32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_multiply %[[VAL2]], %[[MULTIPLIER]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[HALF]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_shift_right_arithmetic %[[VAL4]], %[[TOTAL_SHIFT]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[OUTPUT_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[LHS_32_REQ:.*]] = chlo.broadcast_add %[[VAL5]], %[[OUTPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>

  // CHECK-DAG: %[[RHS_32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = mhlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>, tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// -----

// CHECK-LABEL: @uniform_quantize_add_different_rhs_type
func.func @uniform_quantize_add_different_rhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>

  // CHECK: %[[VAL0:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[INPUT_ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[INPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[MULTIPLIER:.*]] = mhlo.constant dense<16384> : tensor<i32>
  // CHECK-DAG: %[[TOTAL_SHIFT:.*]] = mhlo.constant dense<13> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<4096> : tensor<i32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_multiply %[[VAL2]], %[[MULTIPLIER]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[HALF]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_shift_right_arithmetic %[[VAL4]], %[[TOTAL_SHIFT]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[OUTPUT_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[RHS_32_REQ:.*]] = chlo.broadcast_add %[[VAL5]], %[[OUTPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = mhlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// CHECK-LABEL: @uniform_quantize_add_different_res_type
func.func @uniform_quantize_add_different_res_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[INPUT_ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[INPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[MULTIPLIER:.*]] = mhlo.constant dense<16384> : tensor<i32>
  // CHECK-DAG: %[[TOTAL_SHIFT:.*]] = mhlo.constant dense<13> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<4096> : tensor<i32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_multiply %[[VAL2]], %[[MULTIPLIER]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[HALF]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_shift_right_arithmetic %[[VAL4]], %[[TOTAL_SHIFT]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[OUTPUT_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[LHS_32_REQ:.*]] = chlo.broadcast_add %[[VAL5]], %[[OUTPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>

  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[INPUT_ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL7:.*]] = chlo.broadcast_subtract %[[VAL6]], %[[INPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[MULTIPLIER:.*]] = mhlo.constant dense<16384> : tensor<i32>
  // CHECK-DAG: %[[TOTAL_SHIFT:.*]] = mhlo.constant dense<13> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<4096> : tensor<i32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_multiply %[[VAL7]], %[[MULTIPLIER]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_add %[[VAL8]], %[[HALF]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = chlo.broadcast_shift_right_arithmetic %[[VAL9]], %[[TOTAL_SHIFT]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[OUTPUT_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[RHS_32_REQ:.*]] = chlo.broadcast_add %[[VAL10]], %[[OUTPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL11:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL12:.*]] = chlo.broadcast_subtract %[[VAL11:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL13:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL12:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL14:.*]] = mhlo.convert %[[VAL13:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_requantize_and_dequantize
func.func @uniform_quantize_requantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[INPUT_ZPS:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[INPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[MULTIPLIER:.*]] = mhlo.constant dense<16384> : tensor<i32>
  // CHECK-DAG: %[[TOTAL_SHIFT:.*]] = mhlo.constant dense<13> : tensor<i32>
  // CHECK-DAG: %[[HALF:.*]] = mhlo.constant dense<4096> : tensor<i32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_multiply %[[VAL2]], %[[MULTIPLIER]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL3]], %[[HALF]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_shift_right_arithmetic %[[VAL4]], %[[TOTAL_SHIFT]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[OUTPUT_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_add %[[VAL5]], %[[OUTPUT_ZPS]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL7:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL6]], %[[QUANT_MAX]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %1 = mhlo.uniform_quantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_dot_dequantize
func.func @uniform_quantize_dot_dequantize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[VAL2:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = mhlo.convert %[[VAL4:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL7:.*]] = chlo.broadcast_subtract %[[VAL5]], %[[VAL6:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL8:.*]] = "mhlo.dot"(%[[VAL3]], %[[VAL7]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL10:.*]] = chlo.broadcast_multiply %[[VAL8]], %[[VAL9:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL12:.*]] = chlo.broadcast_add %[[VAL10]], %[[VAL11:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL13:.*]] = mhlo.floor %[[VAL12]] : tensor<?x?xf32>
  // CHECK: %[[VAL15:.*]] = chlo.broadcast_add %[[VAL13]], %[[VAL14:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL16:.*]] = mhlo.convert %[[VAL15]] : (tensor<?x?xf32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL19:.*]] = mhlo.clamp %[[VAL17:.*]], %[[VAL16]], %[[VAL18:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL20:.*]] = mhlo.convert %[[VAL19]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = "mhlo.dot" (%0, %1) : (tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %3 = mhlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_dot_int4
func.func @uniform_quantize_dot_int4(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL2:.*]] = "mhlo.dot"(%[[VAL0:.*]], %[[VAL1:.*]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = mhlo.convert %[[VAL3:.*]] : (tensor<?x?xf32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<-8> : tensor<i32>
  // CHECK-DAG: %[[VAL6:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[VAL7:.*]] = mhlo.clamp %[[VAL5]], %[[VAL4]], %[[VAL6]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %2 = "mhlo.dot" (%0, %1): (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantized_convolution
func.func @uniform_quantized_convolution(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:1>>

  // CHECK: %[[VAL28:.*]] = mhlo.convert %[[VAL12:.*]] : (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[LHS:.*]] = chlo.broadcast_subtract %[[VAL28]], %[[VAL26:.*]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[VAL30:.*]] = mhlo.convert %[[VAL25:.*]] : (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[RHS:.*]] = chlo.broadcast_subtract %[[VAL30]], %[[VAL27:.*]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[VAL32:.*]] = mhlo.convolution(%[[LHS]], %[[RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[VAL43:.*]] = mhlo.clamp %[[VAL41:.*]], %[[VAL40:.*]], %[[VAL42:.*]] : (tensor<i32>, tensor<?x?x?x?xi32>, tensor<i32>) -> tensor<?x?x?x?xi32>
  // CHECK: %[[VAL44:.*]] = mhlo.convert %[[VAL43]] : tensor<?x?x?x?xi32>
  %2 = mhlo.convolution(%0, %1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 2], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [2, 2]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:1>>)
    -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantized_convolution_static_shape
func.func @uniform_quantized_convolution_static_shape(%arg0: tensor<128x28x28x1xf32>, %arg1: tensor<3x3x1x128xf32>) {
  // CHECK: %[[VAL28:.*]] = mhlo.convert %[[VAL12:.*]] : (tensor<128x28x28x1xi8>) -> tensor<128x28x28x1xf32>
  // CHECK: %[[LHS:.*]] = chlo.broadcast_subtract %[[VAL28]], %[[VAL26:.*]] : (tensor<128x28x28x1xf32>, tensor<f32>) -> tensor<128x28x28x1xf32>
  // CHECK: %[[VAL30:.*]] = mhlo.convert %[[VAL25:.*]] : (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xf32>
  // CHECK: %[[RHS:.*]] = chlo.broadcast_subtract %[[VAL30]], %[[VAL27:.*]] : (tensor<3x3x1x128xf32>, tensor<f32>) -> tensor<3x3x1x128xf32>
  // CHECK: %[[VAL32:.*]] = mhlo.convolution(%[[LHS]], %[[RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<128x28x28x1xf32>, tensor<3x3x1x128xf32>) -> tensor<128x26x26x128xf32>
  // CHECK: %[[VAL43:.*]] = mhlo.clamp %[[VAL41:.*]], %[[VAL40:.*]], %[[VAL42:.*]] : (tensor<i32>, tensor<128x26x26x128xi32>, tensor<i32>) -> tensor<128x26x26x128xi32>
  // CHECK: %[[VAL44:.*]] = mhlo.convert %[[VAL43]] : tensor<128x26x26x128xi32>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<128x28x28x1xf32>) -> tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<3x3x1x128xf32>) -> tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>
  %2 = mhlo.convolution(%0, %1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_dot_hybrid
func.func @uniform_quantize_dot_hybrid(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[VAL2:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_multiply %[[VAL3]], %[[VAL4:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL7:.*]] = "mhlo.dot"(%[[VAL6:.*]], %[[VAL5]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_add %[[VAL7]], %[[VAL8:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL10:.*]] = mhlo.floor %[[VAL9]] : tensor<?x?xf32>
  %1 = "mhlo.dot" (%arg0, %0): (tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

func.func @uniform_quantize_dot_hybrid_result_type_not_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) {
  %0 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // expected-error@+2 {{Unsupported result element type for mhlo.dot}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot' that was explicitly marked illegal}}
  %1 = "mhlo.dot" (%arg0, %0): (tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @mhlo_constant_uniform_quantized
func.func @mhlo_constant_uniform_quantized() -> tensor<1xf32> {
  // CHECK: mhlo.constant dense<9> : tensor<1xi8>
  %0 = mhlo.constant() {value = dense<9> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @mhlo_constant_int
func.func @mhlo_constant_int() -> tensor<i32> {
  // CHECK: mhlo.constant dense<-128> : tensor<i32>
  %0 = mhlo.constant() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}
