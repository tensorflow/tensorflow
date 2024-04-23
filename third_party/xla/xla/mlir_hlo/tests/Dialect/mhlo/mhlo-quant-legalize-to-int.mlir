// RUN:  mlir-hlo-opt --mhlo-quant-legalize-to-int -split-input-file %s -verify-diagnostics | FileCheck %s

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

  // CHECK: %[[VAL5:.*]] = mhlo.bitcast_convert %[[VAL4]] : (tensor<?x?xi8>) -> tensor<?x?xi8>
  %1 = mhlo.bitcast_convert %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xi8>

  // CHECK: %[[VAL6:.*]] = mhlo.bitcast_convert %[[VAL5]] : (tensor<?x?xi8>) -> tensor<?x?xi8>
  %2 = mhlo.bitcast_convert %1 : (tensor<?x?xi8>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>

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

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK: #[[$SV:.*]] = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #SV>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?xf32, #[[$SV]]>) -> tensor<?xi8, #[[$SV]]>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #SV>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #SV>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?xf32, #[[$SV]]>, tensor<f32>) -> tensor<?xf32, #[[$SV]]>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #SV>) -> tensor<?xf32, #SV>
  return
}

// -----

// CHECK-LABEL: func @quantize_per_channel
func.func @quantize_per_channel(%arg0: tensor<26x26x3x2xf32>
    ) -> tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[1.100000e+00, 1.100000e-01]>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<[-1.000000e+01, 2.000000e+00]>
  // CHECK-DAG: %[[QMIN:.*]] = mhlo.constant dense<-2.14748365E+9> : tensor<f32>
  // CHECK-DAG: %[[QMAX:.*]] = mhlo.constant dense<2.14748365E+9> : tensor<f32>
  // CHECK: %[[DIVIDE:.*]] = chlo.broadcast_divide %arg0, %[[SCALES]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  // CHECK: %[[ADD:.*]] = chlo.broadcast_add %[[DIVIDE]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  // CHECK: %[[CLAMP:.*]] = mhlo.clamp %[[QMIN]], %[[ADD]], %[[QMAX]]
  // CHECK: %[[ROUND:.*]] = mhlo.round_nearest_even %[[CLAMP]]
  // CHECK: %[[RESULT:.*]] = mhlo.convert %[[ROUND]]
  // CHECK-SAME: (tensor<26x26x3x2xf32>) -> tensor<26x26x3x2xi32>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<26x26x3x2xf32>
      ) -> tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
  return %0 : tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
}

// -----

// CHECK-LABEL: func @dequantize_per_channel
func.func @dequantize_per_channel(
    %arg0: tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
  ) -> tensor<26x26x3x2xf32> {
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[1.100000e+00, 1.100000e-01]>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<[-10, 2]> : tensor<2xi32>
  // CHECK: %[[SUBTRACT:.*]] = chlo.broadcast_subtract
  // CHECK-SAME: %[[INPUT:.*]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xi32>, tensor<2xi32>) -> tensor<26x26x3x2xi32>
  // CHECK: %[[FLOAT:.*]] = mhlo.convert %[[SUBTRACT]]
  // CHECK: %[[RESULT:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[FLOAT]], %[[SCALES]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<26x26x3x2xf32>, tensor<2xf32>) -> tensor<26x26x3x2xf32>
  %0 = mhlo.uniform_dequantize %arg0 : (
      tensor<26x26x3x2x!quant.uniform<i32:f32:3, {1.100000e+00:-10, 1.100000e-01:2}>>
    ) -> tensor<26x26x3x2xf32>
  return %0 : tensor<26x26x3x2xf32>
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

// CHECK-LABEL: func @add_per_channel
func.func @add_per_channel(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>> {
  // CHECK: %[[ADD:.*]] = mhlo.add {{.*}} : tensor<?x3x4x2xi32>
  // CHECK: %[[ZPS:.*]] = mhlo.constant dense<[3, 2]> : tensor<2xi32>
  // CHECK: %[[BCAST_SUB:.*]] = chlo.broadcast_subtract %[[ADD]], %[[ZPS]]
  // CHECK-SAME: {broadcast_dimensions = array<i64: 3>}
  // CHECK-SAME: (tensor<?x3x4x2xi32>, tensor<2xi32>) -> tensor<?x3x4x2xi32>
  // CHECK: return %[[BCAST_SUB]] : tensor<?x3x4x2xi32>
  %11 = mhlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5:3,5.8952903030815205E-5:2}>>
}

// -----

// CHECK-LABEL: func @add_per_channel_no_zp
func.func @add_per_channel_no_zp(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // CHECK: %[[ADD:.*]] = mhlo.add {{.*}} : tensor<?x3x4x2xi32>
  // CHECK: return %[[ADD]] : tensor<?x3x4x2xi32>
  %11 = mhlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

func.func @add_per_channel_i8(
    %arg0: tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // expected-error@+2 {{Per-channel quantized AddOp requires i32 storage type}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %11 = mhlo.add %arg0, %arg1 : tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i8:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

func.func @add_per_channel_different_quant_types(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {1.1:2,0.4:-3}>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // expected-error@+2 {{Per-channel quantized AddOp requires the same quantized element type for all operands and results}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %11 = mhlo.add %arg0, %arg1 : (
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {1.1:2,0.4:-3}>>
    ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
}

// -----

func.func @add_per_channel_per_tensor_mix(
    %arg0: tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
    %arg1: tensor<?x3x4x2x!quant.uniform<i32:f32, 1.1:2>>
  ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>> {
  // expected-error@+1 {{'mhlo.add' op requires compatible types for all operands and results}}
  %11 = mhlo.add %arg0, %arg1 : (
      tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>,
      tensor<?x3x4x2x!quant.uniform<i32:f32, 1.1:2>>
    ) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
  return %11 : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {2.9455460163317514E-5,5.8952903030815205E-5}>>
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

// CHECK-LABEL: func @requantize_per_channel
func.func @requantize_per_channel(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = mhlo.constant dense<[2.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = mhlo.constant dense<[-5.000000e+00, -2.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = mhlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = mhlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
}

// -----

// CHECK-LABEL: func @requantize_per_channel_to_per_tensor
func.func @requantize_per_channel_to_per_tensor(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>> {
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = mhlo.constant dense<[-5.000000e+00, -1.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = mhlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = mhlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:1>>
}

// -----

// CHECK-LABEL: func @requantize_per_tensor_to_per_channel
func.func @requantize_per_tensor_to_per_channel(
    %arg0: tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:2>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %arg0 : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[MERGED_SCALE:.*]] = mhlo.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[VAL2:.*]] = chlo.broadcast_multiply %[[VAL1]], %[[MERGED_SCALE]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[MERGED_ZP:.*]] = mhlo.constant dense<[-1.000000e+00, -2.000000e+00]> : tensor<2xf32>
  // CHECK: %[[VAL3:.*]] = chlo.broadcast_add %[[VAL2]], %[[MERGED_ZP]]
  // CHECK-SAME: broadcast_dimensions = array<i64: 1>
  // CHECK-DAG: %[[QUANT_MIN:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-DAG: %[[QUANT_MAX:.*]] = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK: %[[VAL4:.*]] = mhlo.clamp %[[QUANT_MIN]], %[[VAL3]], %[[QUANT_MAX]]
  // CHECK: %[[VAL5:.*]] = mhlo.round_nearest_even %[[VAL4]] : tensor<2x2xf32>
  // CHECK: %[[VAL6:.*]] = mhlo.convert %[[VAL5]] : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %0 = mhlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32, 5.000000e+00:2>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
}

// -----

func.func @requantize_per_channel_change_axis(
    %arg0: tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+01:3, 5.000000e+00:2}>>
  ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>> {
  // expected-error@+2 {{Cannot requantize while changing quantization_axis}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.uniform_quantize' that was explicitly marked illegal}}
  %0 = mhlo.uniform_quantize %arg0 : (
      tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+01:3, 5.000000e+00:2}>>
    ) -> tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32:1, {5.000000e+00:1, 1.000000e+01:-1}>>
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
  // CHECK: %[[DOT:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<?x?xi8>, tensor<?x?xi8>) -> tensor<?x?xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: "mhlo.get_dimension_size"(%[[DOT]])
  // CHECK-SAME: <{dimension = 0 : i64}> : (tensor<?x?xi32>) -> tensor<i32>
  // CHECK: "mhlo.get_dimension_size"(%[[DOT]])
  // CHECK-SAME: <{dimension = 1 : i64}> : (tensor<?x?xi32>) -> tensor<i32>
  // CHECK: %[[DYN_DIMS:.*]] = "mhlo.concatenate"
  // CHECK-SAME: <{dimension = 0 : i64}>
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
  // CHECK-SAME: <{dimension = 0 : i64}> : (tensor<?x2xi8>) -> tensor<i32>
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

// CHECK-LABEL: func @dot_dynamic_batch_dim
func.func @dot_dynamic_batch_dim(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  ) -> tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>> {
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  // CHECK-SAME: (tensor<?x2xi8>, tensor<2x2xi8>) -> tensor<?x2xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [1]
  // CHECK-SAME: (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<0>
  // CHECK-SAME: (tensor<?xi32>, tensor<2xi64>) -> tensor<?x2xi32>

  // CHECK: mhlo.reduce
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi32>, tensor<i32>) -> tensor<2xi32>
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<1>
  // CHECK-SAME: (tensor<2xi32>, tensor<2xi64>) -> tensor<?x2xi32>

  %0 = "mhlo.dot" (%arg0, %arg1) : (
      tensor<?x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
      tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
    ) -> tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
  return %0 : tensor<?x2x!quant.uniform<i32:f32, 1.000000e+00:3>>
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
  // CHECK: %[[DIM_1_1:.*]] = "mhlo.get_dimension_size"(%[[DOT_RES]])
  // CHECK-SAME: {dimension = 0 : i64}
  // CHECK: %[[DIM_1_2:.*]] = mhlo.convert %[[DIM_1_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_1:.*]] = mhlo.reshape %[[DIM_1_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_2:.*]] = mhlo.constant dense<3> : tensor<1xi64>
  // CHECK: %[[DIM_3_1:.*]] = "mhlo.get_dimension_size"(%[[DOT_RES]])
  // CHECK-SAME: {dimension = 2 : i64}
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

// CHECK-LABEL: func @dot_general_per_channel
func.func @dot_general_per_channel(
    %arg0: tensor<?x2x!quant.uniform<i8:f32, 2.0:3>>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.0,4.0}>>
  ) -> tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>> {
  // CHECK: %[[DOT_RES:.*]] = "mhlo.dot_general"
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]>}

  // Zero point offset contribution from RHS tensor * LHS ZP.

  // CHECK: %[[RHS_I32:.*]] = mhlo.convert %arg1 : (tensor<2x2xi8>)
  // CHECK-SAME: -> tensor<2x2xi32>
  // CHECK: %[[RHS_REDUCE_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RHS_REDUCE:.*]] = mhlo.reduce(%[[RHS_I32]] init: %[[RHS_REDUCE_INIT]])
  // CHECK-SAME: applies mhlo.add across dimensions = [0]
  // CHECK-SAME: (tensor<2x2xi32>, tensor<i32>)
  // CHECK-SAME: -> tensor<2xi32>
  // CHECK: %[[RHS_ZP:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[RHS_ZP_CONTRIB:.*]] = chlo.broadcast_multiply
  // CHECK-SAME: %[[RHS_REDUCE]], %[[RHS_ZP]] :
  // CHECK-SAME: (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>

  // Calculate output dynamic dims.
  // CHECK: %[[DIM_1_1:.*]] = "mhlo.get_dimension_size"(%[[DOT_RES]])
  // CHECK-SAME: {dimension = 0 : i64}
  // CHECK: %[[DIM_1_2:.*]] = mhlo.convert %[[DIM_1_1]] : (tensor<i32>) -> tensor<i64>
  // CHECK: %[[DIM_1:.*]] = mhlo.reshape %[[DIM_1_2]] : (tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[DIM_2:.*]] = mhlo.constant dense<2> : tensor<1xi64>
  // CHECK: %[[OUTPUT_DIMS:.*]] = "mhlo.concatenate"
  // CHECK-SAME: %[[DIM_1]], %[[DIM_2]]

  // CHECK: %[[RHS_ZP_BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: (%[[RHS_ZP_CONTRIB]], %[[OUTPUT_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<1>
  // CHECK-SAME: (tensor<2xi32>, tensor<2xi64>) -> tensor<?x2xi32>
  // CHECK: %[[ZPS_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[ZP_TOTAL_2:.*]] = chlo.broadcast_subtract %[[ZPS_INIT]], %[[RHS_ZP_BCAST]]
  // CHECK-SAME: (tensor<i32>, tensor<?x2xi32>) -> tensor<?x2xi32>
  // CHECK: chlo.broadcast_add %[[DOT_RES]], %[[ZP_TOTAL_2]]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<?x2x!quant.uniform<i8:f32, 2.0:3>>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.0,4.0}>>
  ) -> tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>>
  return %0 : tensor<?x2x!quant.uniform<i32:f32:1, {6.0,8.0}>>
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

// CHECK-LABEL: func @conv2d_default_attr
func.func @conv2d_default_attr(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  ) -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>> {
  // CHECK: mhlo.convolution
  // CHECK-NOT: quant.uniform
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
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

// CHECK-LABEL: func @conv2d_per_channel
func.func @conv2d_per_channel(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
  // CHECK: %[[CONV:.*]] = mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
  // CHECK-SAME: window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]],
  // CHECK-SAME: lhs_dilate = [1, 1], rhs_dilate = [1, 1]
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  // CHECK-SAME: (tensor<128x28x28x1xi8>, tensor<3x3x1x2xi8>) -> tensor<128x26x26x2xi32>

  // CHECK: %[[RHS:.*]] = mhlo.convert %arg1 : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xi32>
  // CHECK: %[[REDUCE:.*]] = mhlo.reduce(%[[RHS]]
  // CHECK-SAME: applies mhlo.add across dimensions = [0, 1, 2]
  // CHECK: %[[LHS_ZP:.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK: %[[ZP_OFFSET:.*]] = chlo.broadcast_multiply %[[REDUCE]], %[[LHS_ZP]]
  // CHECK: %[[ZP_OFFSET_BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[ZP_OFFSET]])
  // CHECK: %[[RES_ZP:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[ZP_OFFSET_TOTAL:.*]] = chlo.broadcast_subtract %[[RES_ZP:.*]], %[[ZP_OFFSET_BCAST]]
  // CHECK: chlo.broadcast_add %[[CONV]], %[[ZP_OFFSET_TOTAL]]
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
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
  // expected-error@+2 {{RHS/result UQ type must have zero zp}}
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

func.func @conv2d_per_channel_rhs_zp_not_zero(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:10}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
  // expected-error@+2 {{RHS/result UQ type must have zero zp.}}
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:10}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

func.func @conv2d_per_channel_res_zp_not_zero(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>> {
  // expected-error@+2 {{RHS/result UQ type must have zero zp.}}
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:3}>>
}

// -----

func.func @conv2d_per_channel_rhs_only(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>> {
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32, 4.000000e+00:0>>
}

// -----

func.func @conv2d_per_channel_res_only(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32, 2.000000e+00:0>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32, 2.000000e+00:0>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

func.func @conv2d_per_channel_unsupported_channel(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:2, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>> {
  // expected-error@+2 {{Conv quantized axis must be out channel axis}}
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
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x2x!quant.uniform<i8:f32:2, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

func.func @conv2d_per_channel_rhs_result_scale_ratio_different(
    %arg0: tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>> {
  // expected-error@+2 {{Per-channel quantizated Conv must have same RHS/Result scale ratio for each channel}}
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
    } : (
      tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.200000e+00:0}>>
}

// -----

// CHECK-LABEL: func @dot_hybrid
func.func @dot_hybrid(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32> {
  // CHECK: %[[VAL1:.*]] = mhlo.optimization_barrier %[[VAL0:.*]] : tensor<?x?xi8>
  // CHECK: %[[VAL2:.*]] = mhlo.convert %[[VAL1:.*]] : (tensor<?x?xi8>) -> tensor<?x?xf32>
  // CHECK: %[[VAL4:.*]] = chlo.broadcast_subtract %[[VAL2]], %[[VAL3:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_multiply %[[VAL4]], %[[VAL5:.*]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL7:.*]] = "mhlo.dot"(%arg0, %[[VAL6]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.dot" (%arg0, %arg1): (
      tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dot_general_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x2xi8>
func.func @dot_general_hybrid_per_channel(
    %arg0: tensor<3x2xf32>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00, 4.000000e+00}>>
  ) -> tensor<3x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = mhlo.optimization_barrier %[[ARG1]] : tensor<2x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = mhlo.convert %[[BARRIER]] : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK-NOT: chlo.broadcast_subtract
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[CONVERT]], %[[SCALES]] {broadcast_dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[DOT:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[MUL]])
  // CHECK-SAME: (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[DOT]]

  %0 = "mhlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<3x2xf32>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00, 4.000000e+00}>>
  ) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func @dot_general_hybrid_per_channel_asymmetric
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x2xi8>
func.func @dot_general_hybrid_per_channel_asymmetric(
    %arg0: tensor<3x2xf32>,
    %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00:10, 4.000000e+00:20}>>
  ) -> tensor<3x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = mhlo.optimization_barrier %[[ARG1]] : tensor<2x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<[1.000000e+01, 2.000000e+01]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = mhlo.convert %[[BARRIER]] : (tensor<2x2xi8>) -> tensor<2x2xf32>
  // CHECK: %[[SUB:.*]] = chlo.broadcast_subtract %[[CONVERT]], %[[ZPS]] {broadcast_dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[SUB]], %[[SCALES]] {broadcast_dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[DOT:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[MUL]])
  // CHECK-SAME: (tensor<3x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[DOT]]

  %0 = "mhlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]>} : (
    tensor<3x2xf32>,
    tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {3.000000e+00:10, 4.000000e+00:20}>>
  ) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
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
  // CHECK-DAG: %[[BARRIER:.*]] = mhlo.optimization_barrier %arg1 : tensor<3x3x1x128xi8>
  // CHECK-DAG: %[[ZP:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[SCALE:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK: %[[RHS:.*]] = mhlo.convert %[[BARRIER]] : (tensor<3x3x1x128xi8>) -> tensor<3x3x1x128xf32>
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

// CHECK-LABEL: func @conv2d_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x28x28x1xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x3x1x2xi8>
func.func @conv2d_hybrid_per_channel(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>
  ) -> tensor<128x26x26x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = mhlo.optimization_barrier %[[ARG1]] : tensor<3x3x1x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = mhlo.convert %[[BARRIER]] : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xf32>
  // CHECK-NOT: chlo.broadcast_subtract
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[CONVERT]], %[[SCALES]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[CONV:.*]] = mhlo.convolution(%[[ARG0]], %[[MUL]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  // CHECK: return %[[CONV]]

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
    } : (
      tensor<128x28x28x1xf32>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:0, 1.000000e+00:0}>>)
    -> tensor<128x26x26x2xf32>
  return %0 : tensor<128x26x26x2xf32>
}

// -----

// CHECK-LABEL: func @conv2d_hybrid_per_channel_asymmetric
// CHECK-SAME: %[[ARG0:.*]]: tensor<128x28x28x1xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x3x1x2xi8>
func.func @conv2d_hybrid_per_channel_asymmetric(
    %arg0: tensor<128x28x28x1xf32>,
    %arg1: tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:10, 1.000000e+00:20}>>
  ) -> tensor<128x26x26x2xf32> {
  // CHECK-DAG: %[[BARRIER:.*]] = mhlo.optimization_barrier %[[ARG1]] : tensor<3x3x1x2xi8>
  // CHECK-DAG: %[[SCALES:.*]] = mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK-DAG: %[[ZPS:.*]] = mhlo.constant dense<[1.000000e+01, 2.000000e+01]> : tensor<2xf32>
  // CHECK-DAG: %[[CONVERT:.*]] = mhlo.convert %[[BARRIER]] : (tensor<3x3x1x2xi8>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[SUB:.*]] = chlo.broadcast_subtract %[[CONVERT]], %[[ZPS]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[MUL:.*]] = chlo.broadcast_multiply %[[SUB]], %[[SCALES]] {broadcast_dimensions = array<i64: 3>} : (tensor<3x3x1x2xf32>, tensor<2xf32>) -> tensor<3x3x1x2xf32>
  // CHECK: %[[CONV:.*]] = mhlo.convolution(%[[ARG0]], %[[MUL]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME: {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x28x28x1xf32>, tensor<3x3x1x2xf32>) -> tensor<128x26x26x2xf32>
  // CHECK: return %[[CONV]]

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
    } : (
      tensor<128x28x28x1xf32>,
      tensor<3x3x1x2x!quant.uniform<i8:f32:3, {2.000000e+00:10, 1.000000e+00:20}>>)
    -> tensor<128x26x26x2xf32>
  return %0 : tensor<128x26x26x2xf32>
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

// CHECK-LABEL: func @mhlo_constant_uniform_quantized_per_channel
func.func @mhlo_constant_uniform_quantized_per_channel() -> () {
  // CHECK: mhlo.constant dense<[9, 4]> : tensor<2xi8>
  %0 = mhlo.constant() {value = dense<[9, 4]> : tensor<2xi8>} : ()
      -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e+00:3, 2.000000e+00:-2}>>
  return
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

// CHECK-LABEL: func @broadcast_per_channel
func.func @broadcast_per_channel(
    %arg0: tensor<2x!quant.uniform<i32:f32:0, {4.000000e+00:0, 2.000000e+00:0}>>
  ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>  {
  // CHECK: "mhlo.broadcast_in_dim"
  // CHECK-SAME: broadcast_dimensions = dense<3> : tensor<1xi64>
  // CHECK-SAME: (tensor<2xi32>) -> tensor<128x26x26x2xi32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<3> : tensor<1xi64>}>: (
      tensor<2x!quant.uniform<i32:f32:0, {4.000000e+00:0, 2.000000e+00:0}>>
    ) -> tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
  return %0 : tensor<128x26x26x2x!quant.uniform<i32:f32:3, {4.000000e+00:0, 2.000000e+00:0}>>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast
func.func @dynamic_broadcast(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    %arg1: tensor<3xi32>
  ) -> tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  // CHECK: "mhlo.dynamic_broadcast_in_dim"
  // CHECK-SAME: broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  // CHECK-SAME: (tensor<1x2xi8>, tensor<3xi32>) -> tensor<?x1x2xi8>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
      broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
    } : (
      tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<3xi32>
    ) -> tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<?x1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
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

// CHECK-LABEL: func @max_per_channel
func.func @max_per_channel(
    %arg0: tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  ) -> tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>> {
  // CHECK: mhlo.maximum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "mhlo.maximum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>,
    tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  ) -> tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
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

// CHECK-LABEL: func @min_per_channel
func.func @min_per_channel(
    %arg0: tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  ) -> tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>> {
  // CHECK: mhlo.minimum
  // CHECK-SAME: tensor<1x2xi8>
  %0 = "mhlo.minimum"(%arg0, %arg0) : (
    tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>,
    tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  ) -> tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32:1, {2.000000e+00:3, 1.000000e+00:-2}>>
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

// CHECK-LABEL: func @concatenate
func.func @concatenate(
    %arg0: tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    %arg1: tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>> {
  // CHECK: mhlo.concatenate
  // CHECK-SAME: (tensor<3x2xi8>, tensor<1x2xi8>) -> tensor<4x2xi8>
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{dimension = 0 : i64}> : (
    tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  return %0 : tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
}

// -----

// CHECK-LABEL: func @pad
func.func @pad(
    %arg0: tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    %arg1: tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>> {
  // CHECK: mhlo.pad
  // CHECK-SAME: (tensor<2x3xi8>, tensor<i8>) -> tensor<5x9xi8>
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[2, 1]> : tensor<2xi64>,
    interior_padding = dense<[1, 2]> : tensor<2xi64>
  }: (
    tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>,
    tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  ) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  return %0 : tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
}

// -----

// CHECK-LABEL: func @reshape
func.func @reshape(
    %arg0: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.reshape
  // CHECK-SAME: (tensor<1x3xi8>) -> tensor<3x1xi8>
  %0 = "mhlo.reshape"(%arg0) : (
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @dynamic_reshape
func.func @dynamic_reshape(
    %arg0: tensor<?x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1: tensor<2xi32>
  ) -> tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.dynamic_reshape
  // CHECK-SAME: (tensor<?x3xi8>, tensor<2xi32>) -> tensor<?x1xi8>
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (
    tensor<?x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<2xi32>
  ) -> tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<?x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @select
func.func @select(
    %arg0: tensor<1x3xi1>,
    %arg1: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg2: tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.select
  // CHECK-SAME: tensor<1x3xi8>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (
    tensor<1x3xi1>,
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(
    %arg0: tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.transpose
  // CHECK-SAME: (tensor<3x1xi8>) -> tensor<1x3xi8>
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (
    tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @gather
func.func @gather(
    %arg0: tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1:  tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.gather
  // CHECK-SAME: (tensor<3x4x2xi8>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi8>
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = dense<[1, 2, 2]> : tensor<3xi64>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(
    %arg0: tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.slice
  // CHECK-SAME: (tensor<3x4xi8>) -> tensor<2x2xi8>
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 2]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (
    tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(
    %arg0: tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>,
    %arg1: tensor<i32>,
    %arg2: tensor<i32>
  ) -> tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>> {
  // CHECK: mhlo.dynamic_slice
  // CHECK-SAME: (tensor<?x4xi8>, tensor<i32>, tensor<i32>) -> tensor<1x1xi8>
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = dense<1> : tensor<2xi64>
  } : (
    tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<i32>,
    tensor<i32>
  ) -> tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  return %0 : tensor<1x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
}

// -----

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(
    %arg0: tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  ) -> tensor<i32> {
  // CHECK: mhlo.get_dimension_size
  // CHECK-SAME: (tensor<?x4xi8>) -> tensor<i32>
  %0 = "mhlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (
      tensor<?x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: reduce_window
func.func @reduce_window(
    %arg0: tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>,
    %arg1: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
  ) -> tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>> {
  // CHECK: mhlo.reduce_window
  // CHECK: %[[ARG2:.*]]: tensor<i8>, %[[ARG3:.*]]: tensor<i8>
  // CHECK: %[[MAX:.*]] = mhlo.maximum %[[ARG2]], %[[ARG3]] : tensor<i8>
  // CHECK: mhlo.return %[[MAX]] : tensor<i8>
  // CHECK: (tensor<2x3x10x3xi8>, tensor<i8>) -> tensor<2x3x10x3xi8>
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>, %arg3: tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
    mhlo.return %1 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>
  }) {padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>} : (tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-49>>) -> tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>
  return %0 : tensor<2x3x10x3x!quant.uniform<i8:f32, 3.000000e-01:-49>>
}
