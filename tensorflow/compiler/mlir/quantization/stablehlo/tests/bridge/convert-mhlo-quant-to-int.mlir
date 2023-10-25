// RUN: stablehlo-quant-opt "-convert-mhlo-quant-to-int=legalize-chlo=false" -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @uniform_quantize_and_dequantize
func.func @uniform_quantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = mhlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL5:.*]] = mhlo.convert %[[VAL4]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL5]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_multiply %[[VAL7]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL8]] : tensor<?x?xf32>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_convert_dequantize
func.func @uniform_quantize_convert_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = mhlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK: %[[VAL5:.*]] = mhlo.convert %[[VAL4]] : tensor<?x?xi8>
  %1 = mhlo.convert %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xi8>

  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[VAL5]] : tensor<?x?xi8>
  %2 = mhlo.convert %1 : (tensor<?x?xi8>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_subtract %[[VAL7]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.convert %[[VAL8]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL10:.*]] = chlo.broadcast_multiply %[[VAL9]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL10]] : tensor<?x?xf32>
  %3 = mhlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_int4
func.func @uniform_quantize_and_dequantize_int4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-8.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<7.000000e+00> : tensor<f32>
  // CHECK: %[[VAL0:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL1:.*]] = chlo.broadcast_add %[[VAL0]], %[[ZPS]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL1]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = mhlo.round_nearest_even %[[VAL2]] : tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = mhlo.convert %[[VAL3]] : (tensor<?x?xf32>) -> tensor<?x?xi4>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL5:.*]] = mhlo.convert %[[VAL4]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL5]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL7:.*]] = mhlo.convert %[[VAL6]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL8:.*]] = chlo.broadcast_multiply %[[VAL7]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL8]] : tensor<?x?xf32>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_type_exensions
func.func @uniform_quantize_and_dequantize_type_exensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xi8, #mhlo.type_extensions<bounds = [4, 4]>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>, tensor<f32>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> tensor<?xi8, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>, tensor<f32>) -> tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
  return
}

// -----

// CHECK-LABEL: func @add
func.func @add(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return %0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @add_i32
func.func @add_i32(
    %arg0: tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1:.*]], %[[VAL3:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-NEXT: return
  %2 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %2 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @add_int4
func.func @add_int4(
    %arg0: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>> {
  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1]], %[[VAL3]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4]], %[[VAL5]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %0 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: @add_different_lhs_type
func.func @add_different_lhs_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[LHS:.*]] = mhlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[LHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = mhlo.constant dense<-5.000000e+00>
  // CHECK: %[[LHS_32:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RHS_32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = mhlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %2 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %2 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: @add_different_rhs_type
func.func @add_different_rhs_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[RHS:.*]] = mhlo.convert %arg1 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[RHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = mhlo.constant dense<-5.000000e+00>
  // CHECK: %[[RHS_32:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL7:.*]] = chlo.broadcast_add %[[LHS_32:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL7:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL10:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL9:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL11:.*]] = mhlo.convert %[[VAL10:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// CHECK-LABEL: @add_different_res_type
func.func @add_different_res_type(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[LHS:.*]] = mhlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[LHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = mhlo.constant dense<-5.000000e+00>
  // CHECK: %[[LHS_32_REQ:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[RHS:.*]] = mhlo.convert %arg1 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[MUL:.*]] = chlo.broadcast_multiply %[[RHS]], %[[COMBINED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[COMBINED_ZP:.*]] = mhlo.constant dense<-5.000000e+00>
  // CHECK: %[[RHS_32_REQ:.*]] = chlo.broadcast_add %[[MUL]], %[[COMBINED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK-DAG: %[[RES_ZPS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAL11:.*]] = chlo.broadcast_add %[[LHS_32_REQ:.*]], %[[RHS_32_REQ:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL12:.*]] = chlo.broadcast_subtract %[[VAL11:.*]], %[[RES_ZPS:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-128> : tensor<i32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<127> : tensor<i32>
  // CHECK: %[[VAL13:.*]] = mhlo.clamp %[[QUANT_MIN:.*]], %[[VAL12:.*]], %[[QUANT_MAX:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL14:.*]] = mhlo.convert %[[VAL13:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.add %arg0, %arg1: (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @requantize
func.func @requantize(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[MERGED_ZP:.*]] = mhlo.constant dense<-5.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = mhlo.round_nearest_even %[[VAL4]] : tensor<?x?xf32>
  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[VAL5]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @requantize_merged_zp_zero
func.func @requantize_merged_zp_zero(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01>>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>> {
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %arg0 : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL3:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL2]], %[[QUANT_MAX]] : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = mhlo.round_nearest_even %[[VAL3]] : tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = mhlo.convert %[[VAL4]] : (tensor<?x?xf32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00>>
}

// -----

// CHECK-LABEL: func @dot
func.func @dot(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
               %arg1: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi32>
  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_int4
func.func @dot_int4(
    %arg0: tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>,
    %arg1: tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi4>, tensor<2x2xi4>) -> tensor<2x2xi32>
  %0 = "mhlo.dot" (%arg0, %arg1): (
      tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i4:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic
func.func @dot_dynamic(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:2>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: mhlo.dot_general
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<?x?xi8>, tensor<?x?xi8>) -> tensor<?x?xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: "mhlo.get_dimension_size"
  // CHECK-SAME: {dimension = 0 : i64} : (tensor<?x?xi8>) -> tensor<i32>
  // CHECK: "mhlo.get_dimension_size"
  // CHECK-SAME: {dimension = 1 : i64} : (tensor<?x?xi8>) -> tensor<i32>
  // CHECK: %[[DYN_DIMS:.*]] = "mhlo.concatenate"
  // CHECK-SAME: {dimension = 0 : i64}
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[DYN_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<0>
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: %[[DYN_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<1>
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:2>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_int4
func.func @dot_dynamic_int4(
    %arg0: tensor<?x?x!quant.uniform<i4:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:2>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: mhlo.dot_general
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<?x?xi4>, tensor<?x?xi4>) -> tensor<?x?xi32>
  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<?x?x!quant.uniform<i4:f32, 2.000000e+00:3>>,
      tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:2>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_contracting_dim
func.func @dot_dynamic_contracting_dim(
    %arg0: tensor<2x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<?x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<2x?xi8>, tensor<?x2xi8>) -> tensor<2x2xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<2x?xi32>, tensor<i32>) -> tensor<2xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<2xi32>

  // CHECK: %[[DYNAMIC_DIM_INIT:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[DYNAMIC_DIM:.*]] = "mhlo.get_dimension_size"
  // CHECK-SAME: {dimension = 0 : i64} : (tensor<?x2xi8>) -> tensor<i32>
  // CHECK: %[[DYNAMIC_DIM_TOTAL:.*]] = mhlo.multiply
  // CHECK-SAME: %[[DYNAMIC_DIM_INIT]], %[[DYNAMIC_DIM]]
  // CHECK: %[[DIMS:.*]] = mhlo.constant dense<9> : tensor<i32>
  // CHECK: %[[DIMS_1:.*]] = mhlo.multiply %[[DIMS]], %[[DYNAMIC_DIM_TOTAL]]
  // CHECK: chlo.broadcast_subtract %[[ZP_OFFSET:.*]], %[[DIMS:.*]]
  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<2x?x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<?x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_dynamic_result_dim
func.func @dot_dynamic_result_dim(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<2x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<?x2xi8>, tensor<2x?xi8>) -> tensor<?x?xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<0>
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<1>
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x?xi32>


  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i32:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:5>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [2]
  // CHECK-SAME: lhs_contracting_dimensions = [2]
  // CHECK-SAME: rhs_contracting_dimensions = [0]

  // Zero point offset contribution from LHS tensor * RHS ZP.

  // CHECK: %[[LHS_I32:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<2x5x6xi8>)
  // CHECK-SAME: -> tensor<2x5x6xi32>
  // CHECK: %[[LHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[LHS_REDUCE:.*]] = mhlo.reduce(%[[LHS_I32]] init: %[[LHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [2]
  // CHECK-SAME: (tensor<2x5x6xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2x5xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[LHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[LHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2x5xi32>, tensor<i32>) -> tensor<2x5xi32>
  // CHECK: %[[LHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[LHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]>
  // CHECK-SAME: (tensor<2x5xi32>) -> tensor<2x5x8xi32>

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[RHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[2, 0]>
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>
  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.add %[[LHS_ZP_BCAST]], %[[RHS_ZP_BCAST]]

  // Zero point offset contribution from LHS ZP * RHS ZP.

  // CHECK: %[[ZPS:.*]] = mhlo.constant dense<90> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZP_TOTAL_1]], %[[ZPS]]
  // CHECK-SAME: (tensor<2x5x8xi32>, tensor<i32>) -> tensor<2x5x8xi32>

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_3:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_5:.*]] = mhlo.convert %[[ZP_TOTAL_4]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_6:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_5]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_6]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:5>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_combined_scale_1
func.func @dot_general_combined_scale_1(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:5>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [2]
  // CHECK-SAME: lhs_contracting_dimensions = [2]
  // CHECK-SAME: rhs_contracting_dimensions = [0]

  // Zero point offset contribution from LHS tensor * RHS ZP.

  // CHECK: %[[LHS_I32:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<2x5x6xi8>)
  // CHECK-SAME: -> tensor<2x5x6xi32>
  // CHECK: %[[LHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[LHS_REDUCE:.*]] = mhlo.reduce(%[[LHS_I32]] init: %[[LHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [2]
  // CHECK-SAME: (tensor<2x5x6xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2x5xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[LHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[LHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2x5xi32>, tensor<i32>) -> tensor<2x5xi32>
  // CHECK: %[[LHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[LHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]>
  // CHECK-SAME: (tensor<2x5xi32>) -> tensor<2x5x8xi32>

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[RHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[2, 0]>
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>
  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.add %[[LHS_ZP_BCAST]], %[[RHS_ZP_BCAST]]

  // Zero point offset contribution from LHS ZP * RHS ZP.

  // CHECK: %[[ZPS:.*]] = mhlo.constant dense<90> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZP_TOTAL_1]], %[[ZPS]]
  // CHECK-SAME: (tensor<2x5x8xi32>, tensor<i32>) -> tensor<2x5x8xi32>

  // Combine dot result with zero point offset and output final result.
  // Do not multiply by combined scale since it is 1.0 and thus no-op.

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_3:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[DOT_RES]], %[[ZP_TOTAL_3]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:5>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 6.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_multiple_batching_dims
func.func @dot_general_multiple_batching_dims(
    %arg0: tensor<2x5x3x7x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x2x7x8x3x!quant.uniform<i8:f32, 1.000000e+00:5>>
  ) -> tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0, 2]
  // CHECK-SAME: rhs_batching_dimensions = [1, 4]
  // CHECK-SAME: lhs_contracting_dimensions = [4, 3]
  // CHECK-SAME: rhs_contracting_dimensions = [0, 2]>}

  // Zero point offset contribution from LHS tensor * RHS ZP.

  // CHECK: %[[LHS_I32:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<2x5x3x7x6xi8>)
  // CHECK-SAME: -> tensor<2x5x3x7x6xi32>
  // CHECK: %[[LHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[LHS_REDUCE:.*]] = mhlo.reduce(%[[LHS_I32]] init: %[[LHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [4, 3]
  // CHECK-SAME: (tensor<2x5x3x7x6xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2x5x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[LHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[LHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2x5x3xi32>, tensor<i32>) -> tensor<2x5x3xi32>
  // CHECK: %[[LHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[LHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 2, 1]>
  // CHECK-SAME: (tensor<2x5x3xi32>) -> tensor<2x3x5x8xi32>

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<6x2x7x8x3xi8>)
  // CHECK-SAME: -> tensor<6x2x7x8x3xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 2]
  // CHECK-SAME: (tensor<6x2x7x8x3xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2x8x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2x8x3xi32>, tensor<i32>) -> tensor<2x8x3xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[RHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 3, 1]>
  // CHECK-SAME: (tensor<2x8x3xi32>) -> tensor<2x3x5x8xi32>
  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.add %[[LHS_ZP_BCAST]], %[[RHS_ZP_BCAST]]

  // Zero point offset contribution from LHS ZP * RHS ZP.

  // CHECK: %[[ZPS:.*]] = mhlo.constant dense<630> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZP_TOTAL_1]], %[[ZPS]]
  // CHECK-SAME: (tensor<2x3x5x8xi32>, tensor<i32>) -> tensor<2x3x5x8xi32>

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x3x5x8xf32>) -> tensor<2x3x5x8xi32>

  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xf32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_3:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_5:.*]] = mhlo.convert %[[ZP_TOTAL_4]]
  // CHECK-SAME: (tensor<2x3x5x8xf32>) -> tensor<2x3x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_6:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_5]]
  // CHECK-SAME: (tensor<i32>, tensor<2x3x5x8xi32>) -> tensor<2x3x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_6]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [1, 4],
      lhs_contracting_dimensions = [4, 3],
      rhs_contracting_dimensions = [0, 2]
    >} : (
      tensor<2x5x3x7x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x2x7x8x3x!quant.uniform<i8:f32, 1.000000e+00:5>>
    ) -> tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x3x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_rhs_zero_zp
func.func @dot_general_rhs_zero_zp(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [2]
  // CHECK-SAME: lhs_contracting_dimensions = [2]
  // CHECK-SAME: rhs_contracting_dimensions = [0]

  // Zero point offset contribution from LHS tensor * RHS ZP is 0 and skipped.

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<6x8x2xi8>)
  // CHECK-SAME: -> tensor<6x8x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<6x8x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<8x2xi32>, tensor<i32>) -> tensor<8x2xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[RHS_ZP_CONTRIB]])
  // CHECK-SAME: broadcast_dimensions = dense<[2, 0]>
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<2x5x8xi32>

  // Zero point offset contribution from LHS ZP * RHS ZP is 0 and skipped.

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_zero_zp
func.func @dot_general_zero_zp(
    %arg0: tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:0>>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [2]
  // CHECK-SAME: lhs_contracting_dimensions = [2]
  // CHECK-SAME: rhs_contracting_dimensions = [0]

  // Both LHS/RHS have zero zp. No zp contribution.

  // CHECK-DAG: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<1.500000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[DOT_RES]] :
  // CHECK-SAME: (tensor<2x5x8xi32>) -> tensor<2x5x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<2x5x8xf32>) -> tensor<2x5x8xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[RES_ZP]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6x!quant.uniform<i8:f32, 2.000000e+00:0>>,
      tensor<6x8x2x!quant.uniform<i8:f32, 3.000000e+00:0>>
    ) -> tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<2x5x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @dot_general_multiple_dynamic_dims
func.func @dot_general_multiple_dynamic_dims(
    %arg0: tensor<?x?x3x?x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<6x?x?x8x3x!quant.uniform<i8:f32, 1.000000e+00:5>>
  ) -> tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_batching_dimensions = [0, 2]
  // CHECK-SAME: rhs_batching_dimensions = [1, 4]
  // CHECK-SAME: lhs_contracting_dimensions = [4, 3]
  // CHECK-SAME: rhs_contracting_dimensions = [0, 2]>}

  // Zero point offset contribution from LHS tensor * RHS ZP.

  // CHECK: %[[LHS_I32:.*]] = mhlo.convert %[[LHS:.*]] : (tensor<?x?x3x?x6xi8>)
  // CHECK-SAME: -> tensor<?x?x3x?x6xi32>
  // CHECK: %[[LHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[LHS_REDUCE:.*]] = mhlo.reduce(%[[LHS_I32]] init: %[[LHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [4, 3]
  // CHECK-SAME: (tensor<?x?x3x?x6xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<?x?x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[LHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[LHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<?x?x3xi32>, tensor<i32>) -> tensor<?x?x3xi32>

  // Calculate output dynamic dims.
  // CHECK: %[[DIM_1_1:.*]] = "mhlo.get_dimension_size"(%[[LHS]])
  // CHECK-SAME: {dimension = 0 : i64}
  // CHECK: %[[DIM_1_2:.*]] = mhlo.convert %[[DIM_1_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_1:.*]] = mhlo.reshape %[[DIM_1_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_2:.*]] = mhlo.constant dense<3> : tensor<1xi64>
  // CHECK: %[[DIM_3_1:.*]] = "mhlo.get_dimension_size"(%[[LHS]])
  // CHECK-SAME: {dimension = 1 : i64}
  // CHECK: %[[DIM_3_2:.*]] = mhlo.convert %[[DIM_3_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_3:.*]] = mhlo.reshape %[[DIM_3_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_4:.*]] = mhlo.constant dense<8> : tensor<1xi64>
  // CHECK: %[[OUTPUT_DIMS:.*]] = "mhlo.concatenate"
  // CHECK-SAME: %[[DIM_1]], %[[DIM_2]], %[[DIM_3]], %[[DIM_4]]

  // CHECK: %[[LHS_ZP_BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[LHS_ZP_CONTRIB]], %[[OUTPUT_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 2, 1]>
  // CHECK-SAME: (tensor<?x?x3xi32>, tensor<4xi64>) -> tensor<?x3x?x8xi32>

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS:.*]] : (tensor<6x?x?x8x3xi8>)
  // CHECK-SAME: -> tensor<6x?x?x8x3xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 2]
  // CHECK-SAME: (tensor<6x?x?x8x3xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<?x8x3xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<?x8x3xi32>, tensor<i32>) -> tensor<?x8x3xi32>

  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[RHS_ZP_CONTRIB]], %[[OUTPUT_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 3, 1]>
  // CHECK-SAME: (tensor<?x8x3xi32>, tensor<4xi64>) -> tensor<?x3x?x8xi32>
  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.add %[[LHS_ZP_BCAST]], %[[RHS_ZP_BCAST]]

  // Zero point offset contribution from LHS ZP * RHS ZP.

  // CHECK: %[[ZPS_INIT:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %[[DYN_DIM:.*]] = "mhlo.get_dimension_size"(%[[RHS]])
  // CHECK: %[[ZPS_1:.*]] = mhlo.multiply %[[ZPS_INIT]], %[[DYN_DIM]]
  // CHECK: %[[STATIC_DIM:.*]] = mhlo.constant dense<90> : tensor<i32>
  // CHECK: %[[ZPS:.*]] = mhlo.multiply %[[STATIC_DIM]], %[[ZPS_1]]
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZP_TOTAL_1]], %[[ZPS]]
  // CHECK-SAME: (tensor<?x3x?x8xi32>, tensor<i32>) -> tensor<?x3x?x8xi32>

  // Combine dot result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[DOT_RES]]
  // CHECK-SAME: (tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<?x3x?x8xf32>) -> tensor<?x3x?x8xi32>

  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xf32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_3:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_5:.*]] = mhlo.convert %[[ZP_TOTAL_4]]
  // CHECK-SAME: (tensor<?x3x?x8xf32>) -> tensor<?x3x?x8xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<7> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_6:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_5]]
  // CHECK-SAME: (tensor<i32>, tensor<?x3x?x8xi32>) -> tensor<?x3x?x8xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_6]]

  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [1, 4],
      lhs_contracting_dimensions = [4, 3],
      rhs_contracting_dimensions = [0, 2]
    >} : (
      tensor<?x?x3x?x6x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<6x?x?x8x3x!quant.uniform<i8:f32, 1.000000e+00:5>>
    ) -> tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
  return %0 : tensor<?x3x?x8x!quant.uniform<i32:f32, 4.000000e+00:7>>
}

// -----

// CHECK-LABEL: func @conv2d_dynamic
func.func @conv2d_dynamic(
    %arg0: tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK-NOT: mhlo.pad

  // CHECK: %[[CONV:.*]] = mhlo.convolution
  // CHECK-SAME: (%[[LHS:.*]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 2], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<?x?x?x?xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<?xi32>
  // CHECK: %[[LHS_ZP:.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK-SAME: (tensor<?xi32>, tensor<4xi64>) -> tensor<?x?x?x?xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 2], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [2, 2]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_static
func.func @conv2d_static(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK-NOT: mhlo.pad

  // CHECK: %[[CONV:.*]] = mhlo.convolution
  // CHECK-SAME: (%[[LHS:.*]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<3x3x1x128xi8>) -> tensor<128x26x26x128xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<3x3x1x128xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<128xi32>
  // CHECK: %[[LHS_ZP:.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<128xi32>, tensor<i32>) -> tensor<128xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK-SAME: (tensor<128xi32>) -> tensor<128x26x26x128xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<128x26x26x128xf32>) -> tensor<128x26x26x128xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<128x26x26x128xf32>) -> tensor<128x26x26x128xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<128x26x26x128xi32>) -> tensor<128x26x26x128xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv2d_static_padding
func.func @conv2d_static_padding(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // Explicitly pad LHS with ZP.

  // CHECK: %[[LHS_ZP_i8:.*]] = mhlo.constant dense<4> : tensor<i8>
  // CHECK: %[[LHS_PAD:.*]] = "mhlo.pad"(%[[LHS:.*]], %[[LHS_ZP_i8]])
  // CHECK-SAME: edge_padding_high = dense<[0, 2, 4, 0]>
  // CHECK-SAME: edge_padding_low = dense<[0, 1, 3, 0]>
  // CHECK-SAME: interior_padding = dense<0>
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<i8>) -> tensor<128x31x35x1xi8>

  // Convolution with padding removed.

  // CHECK: %[[CONV:.*]] = mhlo.convolution
  // CHECK-SAME: (%[[LHS_PAD]], %[[RHS:.{1,4}]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x31x35x1xi8>, tensor<3x3x1x128xi8>) -> tensor<128x29x33x128xi32>

  // Zero point offset contribution from LHS ZP * RHS.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %[[RHS]]
  // CHECK-SAME: (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xi32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]]
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 1, 2]
  // CHECK-SAME: (tensor<3x3x1x128xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<128xi32>
  // CHECK: %[[LHS_ZP:.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply %[[RHS_REDUCE]], %[[LHS_ZP]]
  // CHECK-SAME: (tensor<128xi32>, tensor<i32>) -> tensor<128xi32>
  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.broadcast_in_dim"
  // CHECK-SAME: %[[RHS_ZP_CONTRIB]]
  // CHECK-SAME: {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK-SAME: (tensor<128xi32>) -> tensor<128x29x33x128xi32>

  // Combine conv result with zero point offset and output final result.

  // CHECK: %[[COMBINED_SCALE:.*]] = mhlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[RES_FP:.*]] = mhlo.convert %[[CONV]]
  // CHECK-SAME: (tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xf32>
  // CHECK: %[[RES_FP_1:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RES_FP:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[RES_INT:.*]] = mhlo.convert %[[RES_FP_1]]
  // CHECK-SAME: (tensor<128x29x33x128xf32>) -> tensor<128x29x33x128xi32>

  // CHECK: %[[ZP_TOTAL_1:.*]] = mhlo.convert %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xf32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[ZP_TOTAL_1:.*]], %[[COMBINED_SCALE]]
  // CHECK: %[[ZP_TOTAL_3:.*]] = mhlo.convert %[[ZP_TOTAL_2]]
  // CHECK-SAME: (tensor<128x29x33x128xf32>) -> tensor<128x29x33x128xi32>

  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<5> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_4:.*]] = chlo.broadcast_subtract %[[RES_ZP]], %[[ZP_TOTAL_3]]
  // CHECK-SAME: (tensor<i32>, tensor<128x29x33x128xi32>) -> tensor<128x29x33x128xi32>
  // CHECK: chlo.broadcast_add %[[RES_INT]], %[[ZP_TOTAL_4]]
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[1, 2], [3, 4]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x29x33x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

// CHECK-LABEL: func @conv3d_static
func.func @conv3d_static(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>{
  // CHECK-NOT: mhlo.pad

  // CHECK: mhlo.convolution
  // CHECK-SAME: dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
  // CHECK-SAME: window = {stride = [1, 1, 1], pad = {{\[}}[0, 0], [0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1, 1], rhs_dilate = [1, 1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x28x1xi8>, tensor<3x3x3x1x128xi8>) -> tensor<128x26x26x26x128xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 1, 2, 3]
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return %0 : tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
}

// -----

func.func @conv3d_rhs_zp_not_zero(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:-2>>) {
  // expected-error@+2 {{RHS UQ type must have zero zp}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:-2>>)
    -> tensor<128x26x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv3d_rhs_invalid_dilate(
    %arg0: tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // expected-error@+2 {{lhs_dilation must be 1}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [2, 2, 2],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x53x53x53x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv3d_non_nhwc(
    %arg0: tensor<128x1x28x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // expected-error@+2 {{Convolution data format must be NHWC}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1, 2]x[0, 1, 2, i, o]->[b, f, 0, 1, 2],
    window = {
      stride = [1, 1, 1], pad = [[0, 0], [0, 0], [0, 0]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x1x28x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x128x26x26x26x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @conv2d_non_nhwc(
    %arg0: tensor<128x1x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // expected-error@+2 {{Convolution data format must be NHWC}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x1x28x28x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x128x26x26x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

// CHECK-LABEL: func @dot_hybrid
func.func @dot_hybrid(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32> {
  // CHECK: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_subtract %[[VAL1]], %[[VAL2:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL5:.*]] = chlo.broadcast_multiply %[[VAL3]], %[[VAL4:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL7:.*]] = "mhlo.dot"(%arg0, %[[VAL5]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.dot" (%arg0, %arg1): (
      tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

func.func @dot_hybrid_result_type_not_float(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) {
  // expected-error@+2 {{Invalid input/output type for Dot/Convolution op}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot' that was explicitly marked illegal}}
  %1 = "mhlo.dot" (%arg0, %arg1): (
      tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

func.func @dot_hybrid_lhs_type_not_float(
    %arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>,
    %arg1: tensor<?x?xf32>) {
  // expected-error@+2 {{Invalid input/output type for Dot/Convolution op}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot' that was explicitly marked illegal}}
  %1 = "mhlo.dot" (%arg0, %arg1): (
      tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?xf32>
    ) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @conv2d_static_hybrid
func.func @conv2d_static_hybrid(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>
  ) -> tensor<128x26x26x128xf32> {
  // CHECK-DAG: %[[ZP:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[SCALE:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK: %[[RHS:.*]] = mhlo.convert %arg1 : (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xf32>
  // CHECK: %[[SUB:.*]] = chlo.broadcast_subtract %[[RHS]], %[[ZP]]
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[SUB]], %[[SCALE]]
  // CHECK: mhlo.convolution(%arg0, %[[MUL]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME: stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]]
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: : (tensor<128x28x28x1xf32>, tensor<3x3x1x128xf32>) -> tensor<128x26x26x128xf32>
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1xf32>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:1>>)
    -> tensor<128x26x26x128xf32>
  return %0 : tensor<128x26x26x128xf32>
}

// -----

func.func @conv2d_hybrid_result_not_float(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>) {
  // expected-error@+2 {{Invalid input/output type for Dot/Convolution op}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    }
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<128x28x28x1xf32>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

func.func @dot_general_hybrid_result_not_float(
    %arg0: tensor<2x5x6xf32>,
    %arg1: tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:5>>) {
  // expected-error@+2 {{Invalid input/output type for Dot/Convolution op}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot_general' that was explicitly marked illegal}}
  %0 = "mhlo.dot_general" (%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6xf32>,
      tensor<6x8x2x!quant.uniform<i8:f32, 1.000000e+00:5>>
    ) -> tensor<2x5x8x!quant.uniform<i8:f32, 4.000000e+00:7>>
  return
}

// -----

// CHECK-LABEL: func @mhlo_constant_uniform_quantized
func.func @mhlo_constant_uniform_quantized() -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>> {
  // CHECK: mhlo.constant dense<9> : tensor<1xi8>
  %0 = mhlo.constant() {value = dense<9> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return %0 : tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @mhlo_constant_int
func.func @mhlo_constant_int() -> tensor<i32> {
  // CHECK: mhlo.constant dense<-128> : tensor<i32>
  %0 = mhlo.constant() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @broadcast
func.func @broadcast(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: "mhlo.broadcast_in_dim"
  // CHECK-SAME: broadcast_dimensions = dense<[2, 0]> : tensor<2xi64>
  // CHECK-SAME: (tensor<1x2xi8>) -> tensor<2x3x1xi8>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<[2, 0]> : tensor<2xi64>
    } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @max
func.func @max(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: mhlo.maximum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "mhlo.maximum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @min
func.func @min(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: mhlo.minimum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "mhlo.minimum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

// CHECK-LABEL: func @function(%arg0: tensor<1x2xi8>) -> tensor<1x2xi8>
func.func @function(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: return %arg0 : tensor<1x2xi8>
  return %arg0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

func.func @min_mix_uq_type1(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<1x2x!quant.uniform<i8:f32, 1.000000e+00:2>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.minimum' that was explicitly marked illegal}}
  %0 = "mhlo.minimum"(%arg0, %arg1) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 1.000000e+00:2>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}

// -----

func.func @min_mix_uq_type2(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e+00:2>> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.minimum' that was explicitly marked illegal}}
  %0 = "mhlo.minimum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e+00:2>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 1.000000e+00:2>>
}
