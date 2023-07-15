// RUN: stablehlo-quant-opt "-convert-mhlo-quant-to-int=legalize-chlo=true" -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @uniform_quantize_and_dequantize
func.func @uniform_quantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_int4
func.func @uniform_quantize_and_dequantize_int4(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i4:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_type_exensions
func.func @uniform_quantize_and_dequantize_type_exensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>, #mhlo.type_extensions<bounds = [4, 4]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [4, 4]>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_sparse_tensor_encoding
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_add_dequantize
func.func @uniform_quantize_add_dequantize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32,1.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %3 = mhlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @uniform_quantize_add_different_lhs_type
func.func @uniform_quantize_add_different_lhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>, tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// -----

// CHECK-LABEL: @uniform_quantize_add_different_rhs_type
func.func @uniform_quantize_add_different_rhs_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// CHECK-LABEL: @uniform_quantize_add_different_res_type
func.func @uniform_quantize_add_different_res_type(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %2 = mhlo.add %0, %1: (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantize_requantize_and_dequantize
func.func @uniform_quantize_requantize_and_dequantize(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>
  %1 = mhlo.uniform_quantize %0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+01:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<?x?x!quant.uniform<i8:f32, 5.000000e+00:1>>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_dot_dequantize
func.func @uniform_quantize_dot_dequantize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %2 = "mhlo.dot" (%0, %1) : (tensor<?x?x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %3 = mhlo.uniform_dequantize %2 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_convolution
func.func @uniform_quantized_convolution(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:1>>
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

// CHECK-LABEL: func @uniform_quantize_dot_hybrid
func.func @uniform_quantize_dot_hybrid(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg1 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  %1 = "mhlo.dot" (%arg0, %0): (tensor<?x?xf32>, tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return
}