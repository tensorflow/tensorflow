// RUN: tf-opt -convert-mhlo-quant-to-int -split-input-file %s -verify-diagnostics | FileCheck %s

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

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL8]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL11:.*]] = chlo.broadcast_multiply %[[VAL10]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL11]] : tensor<?x?xf32>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

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

  // CHECK-DAG: %[[SCALES_DQ:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZPS_DQ:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK: %[[VAL8:.*]] = mhlo.convert %[[VAL7]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = chlo.broadcast_subtract %[[VAL8]], %[[ZPS_DQ]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9]] : (tensor<?x?xi32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL11:.*]] = chlo.broadcast_multiply %[[VAL10]], %[[SCALES_DQ]] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[VAL11]] : tensor<?x?xf32>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_type_exensions
func.func @uniform_quantize_and_dequantize_type_exensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xi8, #mhlo.type_extensions<bounds = [4, 4]>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>, tensor<f32>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> () {
  // CHECK: %[[QUANTIZED:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?xi32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?xi8, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  // CHECK: %[[DEQUANTIZED:.*]] = chlo.broadcast_multiply %[[VAL1:.*]], %[[CONST_SCALE:.*]] : (tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>, tensor<f32>) -> tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add
func.func @uniform_quantize_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi8>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1:.*]], %[[VAL3:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4:.*]], %[[VAL5:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6:.*]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi8>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add_int4
func.func @uniform_quantize_add_int4(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  // CHECK-DAG: %[[VAL1:.*]] = mhlo.convert %[[VAL0:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL3:.*]] = mhlo.convert %[[VAL2:.*]] : (tensor<?x?xi4>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[VAL5:.*]] = mhlo.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[VAL4:.*]] = chlo.broadcast_add %[[VAL1:.*]], %[[VAL3:.*]] : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL6:.*]] = chlo.broadcast_subtract %[[VAL4:.*]], %[[VAL5:.*]] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL9:.*]] = mhlo.clamp %[[VAL7:.*]], %[[VAL6:.*]], %[[VAL8:.*]] : (tensor<i32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  // CHECK: %[[VAL10:.*]] = mhlo.convert %[[VAL9:.*]] : (tensor<?x?xi32>) -> tensor<?x?xi4>
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  return
}

// -----

func.func @uniform_quantize_add_different_lhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // expected-error@+2 {{AddOp requires the same quantized element type for all operands and results}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

func.func @uniform_quantize_add_different_rhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>
  // expected-error@+2 {{AddOp requires the same quantized element type for all operands and results}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return
}

// -----

func.func @uniform_quantize_add_result_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // expected-error@+2 {{AddOp requires the same quantized element type for all operands and results}}
  // expected-error@+1 {{failed to legalize operation 'mhlo.add' that was explicitly marked illegal}}
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_requantize_and_dequantize
func.func @uniform_quantize_requantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
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
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
