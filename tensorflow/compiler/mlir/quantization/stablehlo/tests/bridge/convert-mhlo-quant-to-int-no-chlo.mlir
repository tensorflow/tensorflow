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
func.func @uniform_quantize_and_dequantize_sparse_tensor_encoding(%arg0: tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> () {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
  %1 = mhlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>) -> tensor<?xf32, #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>>
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

// CHECK-LABEL: func @uniform_quantized_conv2d_dynamic
func.func @uniform_quantized_conv2d_dynamic(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>
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
    } : (tensor<?x?x?x?x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<?x?x?x?x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<?x?x?x?x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantized_conv2d_static
func.func @uniform_quantized_conv2d_static(%arg0: tensor<128x28x28x1xf32>, %arg1: tensor<3x3x1x128xf32>) {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<128x28x28x1xf32>) -> tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<3x3x1x128xf32>) -> tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
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
    } : (tensor<128x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>)
    -> tensor<128x26x26x128x!quant.uniform<i32:f32, 1.000000e+00:5>>
  return
}

// -----

// CHECK-LABEL: func @uniform_quantized_conv3d_static
func.func @uniform_quantized_conv3d_static(%arg0: tensor<128x28x28x28x1xf32>, %arg1: tensor<3x3x3x1x128xf32>) {
  // CHECK-NOT: chlo
  %0 = mhlo.uniform_quantize %arg0 : (tensor<128x28x28x28x1xf32>) -> tensor<128x28x28x28x1x!quant.uniform<i8:f32, 2.000000e+00:4>>
  %1 = mhlo.uniform_quantize %arg1 : (tensor<3x3x3x1x128xf32>) -> tensor<3x3x3x1x128x!quant.uniform<i8:f32, 3.000000e+00:0>>
  %2 = mhlo.convolution(%0, %1)
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

// -----

// CHECK-LABEL: func @mhlo_constant_uniform_quantized
func.func @mhlo_constant_uniform_quantized() -> tensor<1xf32> {
  // CHECK-NOT: chlo
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

// -----

// CHECK-LABEL: func @uniform_quantize_broadcast_dequantize
func.func @uniform_quantize_broadcast_dequantize(%arg0: tensor<1x2xf32>) -> tensor<2x3x1xf32> {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>

  // CHECK-NOT: chlo
  %1 = "mhlo.broadcast_in_dim"(%0) {
    broadcast_dimensions = dense<[2, 0]> : tensor<2xi64>
    } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<2x3x1x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<2x3x1xf32>
  return %2 : tensor<2x3x1xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_max_dequantize
func.func @uniform_quantize_max_dequantize(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>

  // CHECK-NOT: chlo
  %1 = "mhlo.maximum"(%0, %0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_min_dequantize
func.func @uniform_quantize_min_dequantize(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = mhlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>

  // CHECK-NOT: chlo
  %1 = "mhlo.minimum"(%0, %0) : (
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  %2 = mhlo.uniform_dequantize %1 : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}
