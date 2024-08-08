// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: main
func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
}

// CHECK: return %arg0 : tensor<5x7xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.transpose
//===----------------------------------------------------------------------===//

// CHECK-LABEL: transpose_2d
func.func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// CHECK-NEXT: %0 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg0, %1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK-NEXT: return %2 : tensor<3x2xf32>

// -----

// CHECK-LABEL: transpose_3d
func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[2, 1, 0]> : tensor<3xi64>}> : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// CHECK-NEXT: %0 = "tfl.pseudo_const"() <{value = dense<[2, 1, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg0, %1) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
// CHECK-NEXT: return %2 : tensor<3x2x1xf32>

// -----

// CHECK-LABEL: transpose_dynamic_2d
func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<?x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// CHECK-NEXT: %0 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg0, %1) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<4x?xf32>
// CHECK-NEXT: return %2 : tensor<4x?xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.dot_general
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dot_general
func.func @dot_general(%arg0: tensor<3x2x6x5x1xf32>, %arg1: tensor<3x2x4x6xf32>) -> tensor<3x5x1x4xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1, 2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1, 3]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<3x2x6x5x1xf32>, tensor<3x2x4x6xf32>) -> tensor<3x5x1x4xf32>
  func.return %0 : tensor<3x5x1x4xf32>
}

// CHECK:     %[[TRANSPOSED_0:.*]] = "tfl.transpose"
// CHECK:     %[[TRANSPOSED_1:.*]] = "tfl.transpose"
// CHECK-NEXT: %[[RESHAPED_0:.*]] = mhlo.reshape %[[TRANSPOSED_0]]
// CHECK-NEXT: %[[RESHAPED_1:.*]] = mhlo.reshape %[[TRANSPOSED_1]]
// CHECK-NEXT: %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<3x5x12xf32>, tensor<3x12x4xf32>) -> tensor<3x5x4xf32>
// CHECK-NEXT: %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT: return %[[RESHAPED_BMM]] : tensor<3x5x1x4xf32>

// -----

// CHECK-LABEL: dot_general_repeated
func.func @dot_general_repeated(%arg0: tensor<1x1x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1x1x1024xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1x1024xf32>
  func.return %0 : tensor<1x1x1024xf32>
}

// CHECK:     %[[RESHAPED_0:.*]] = mhlo.reshape %arg0
// CHECK-NEXT: %[[RESHAPED_1:.*]] = mhlo.reshape %arg1
// CHECK-NEXT: %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : {{.*}} -> tensor<1x1024xf32>
// CHECK-NEXT: %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT: return %[[RESHAPED_BMM]] : tensor<1x1x1024xf32>

// -----

// CHECK-LABEL: dot_general_int8
func.func @dot_general_int8(%arg0: tensor<256xi8>, %arg1: tensor<256x8xi8>) -> tensor<8xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<256xi8>, tensor<256x8xi8>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// CHECK:      %[[RESHAPED_0:.*]] = mhlo.reshape %arg0
// CHECK-NEXT: %[[RESHAPED_1:.*]] = mhlo.reshape %arg1
// CHECK-NEXT: %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : {{.*}} -> tensor<1x8xi32>
// CHECK-NEXT: %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT: return %[[RESHAPED_BMM]] : tensor<8xi32>

// -----

// CHECK-LABEL: dot_general_dynamic_rhs_out_dim
func.func @dot_general_dynamic_rhs_out_dim(%arg0: tensor<4x4x256xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]
    >} : (tensor<4x4x256xf32>, tensor<4x?x256xf32>) -> tensor<4x4x?xf32>
  func.return %0 : tensor<4x4x?xf32>
}

// CHECK:      %0 = "tfl.pseudo_const"() <{value = dense<[0, 2, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg1, %1) : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT: %3 = mhlo.reshape %arg0 : (tensor<4x4x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT: %4 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %5 = "tfl.pseudo_const"() <{value = dense<[-1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %6 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %7 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %8 = "tfl.unsorted_segment_prod"(%4, %5, %7) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %9 = "tfl.unsorted_segment_prod"(%4, %6, %7) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %10 = "tfl.pseudo_const"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %11 = "tfl.concatenation"(%10, %9, %8) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %12 = mhlo.dynamic_reshape %2, %11 : (tensor<4x256x?xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT: %13 = "tfl.batch_matmul"(%3, %12) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x4x256xf32>, tensor<4x256x?xf32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: %14 = "tfl.shape"(%arg0) : (tensor<4x4x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %15 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %17 = "tfl.gather"(%14, %16) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<3xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %18 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-NEXT: %19 = "tfl.gather"(%15, %18) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<3xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT: %20 = "tfl.concatenation"(%17, %19) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %21 = mhlo.dynamic_reshape %13, %20 : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: return %21 : tensor<4x4x?xf32>

// -----

// CHECK-LABEL: dot_general_dynamic_batch_dim
func.func @dot_general_dynamic_batch_dim(%arg0: tensor<2x?x2x3xf32>, %arg1: tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >} : (tensor<2x?x2x3xf32>, tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32>
  func.return %0 : tensor<2x?x2x4xf32>
}

// CHECK:      %0 = "tfl.pseudo_const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<4xi64>) -> tensor<4xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg1, %1) : (tensor<2x?x4x3xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT: %3 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %4 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %5 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %6 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %9 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %10 = "tfl.gather"(%3, %9) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %11 = "tfl.concatenation"(%10, %7, %8) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT: %12 = mhlo.dynamic_reshape %arg0, %11 : (tensor<2x?x2x3xf32>, tensor<4xi32>) -> tensor<2x?x2x3xf32>
// CHECK-NEXT: %13 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %14 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %15 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %17 = "tfl.unsorted_segment_prod"(%13, %14, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %18 = "tfl.unsorted_segment_prod"(%13, %15, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %19 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %20 = "tfl.gather"(%13, %19) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %21 = "tfl.concatenation"(%20, %18, %17) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT: %22 = mhlo.dynamic_reshape %2, %21 : (tensor<2x?x3x4xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT: %23 = "tfl.batch_matmul"(%12, %22) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x?x2x3xf32>, tensor<2x?x3x4xf32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT: %24 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %25 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %26 = "tfl.pseudo_const"() <{value = dense<[0, 1, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %27 = "tfl.gather"(%24, %26) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %28 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-NEXT: %29 = "tfl.gather"(%25, %28) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT: %30 = "tfl.concatenation"(%27, %29) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT: %31 = mhlo.dynamic_reshape %23, %30 : (tensor<2x?x2x4xf32>, tensor<4xi32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT: return %31 : tensor<2x?x2x4xf32>

// -----


// CHECK-LABEL: dot_general_dynamic_lhs_rhs_out_dims
func.func @dot_general_dynamic_lhs_rhs_out_dims(%arg0: tensor<2x2x?x3xf32>, %arg1: tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]
    >} : (tensor<2x2x?x3xf32>, tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32>
  func.return %0 : tensor<2x2x?x4x?xf32>
}

// CHECK:      %0 = "tfl.pseudo_const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-NEXT: %1 = "tfl.cast"(%0) : (tensor<4xi64>) -> tensor<4xi32>
// CHECK-NEXT: %2 = "tfl.transpose"(%arg1, %1) : (tensor<2x4x?x3xf32>, tensor<4xi32>) -> tensor<2x3x4x?xf32>
// CHECK-NEXT: %3 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %4 = "tfl.pseudo_const"() <{value = dense<[-1, 0, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %5 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %6 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %9 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %10 = "tfl.concatenation"(%9, %7, %8) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %11 = mhlo.dynamic_reshape %arg0, %10 : (tensor<2x2x?x3xf32>, tensor<3xi32>) -> tensor<2x?x3xf32>
// CHECK-NEXT: %12 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %13 = "tfl.pseudo_const"() <{value = dense<[-1, 0, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %14 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %15 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %16 = "tfl.unsorted_segment_prod"(%12, %13, %15) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %17 = "tfl.unsorted_segment_prod"(%12, %14, %15) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %18 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %19 = "tfl.concatenation"(%18, %17, %16) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %20 = mhlo.dynamic_reshape %2, %19 : (tensor<2x3x4x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK-NEXT: %21 = "tfl.batch_matmul"(%11, %20) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x?x3xf32>, tensor<2x3x?xf32>) -> tensor<2x?x?xf32>
// CHECK-NEXT: %22 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %23 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %24 = "tfl.pseudo_const"() <{value = dense<[0, 1, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %25 = "tfl.gather"(%22, %24) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %26 = "tfl.pseudo_const"() <{value = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %27 = "tfl.gather"(%23, %26) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %28 = "tfl.concatenation"(%25, %27) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<2xi32>) -> tensor<5xi32>
// CHECK-NEXT: %29 = mhlo.dynamic_reshape %21, %28 : (tensor<2x?x?xf32>, tensor<5xi32>) -> tensor<2x2x?x4x?xf32>
// CHECK-NEXT: return %29 : tensor<2x2x?x4x?xf32>

// -----

// CHECK-LABEL: dot_general_dynamic_contracting_dim
func.func @dot_general_dynamic_contracting_dim(%arg0: tensor<4x4x?xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x256xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
  func.return %0 : tensor<4x4x256xf32>
}

// CHECK:      %0 = "tfl.shape"(%arg0) : (tensor<4x4x?xf32>) -> tensor<3xi32>
// CHECK-NEXT: %1 = "tfl.pseudo_const"() <{value = dense<[-1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %2 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %3 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %4 = "tfl.unsorted_segment_prod"(%0, %1, %3) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %5 = "tfl.unsorted_segment_prod"(%0, %2, %3) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %6 = "tfl.pseudo_const"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %7 = "tfl.concatenation"(%6, %4, %5) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %8 = mhlo.dynamic_reshape %arg0, %7 : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: %9 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %10 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %11 = "tfl.pseudo_const"() <{value = dense<[-1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %12 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %13 = "tfl.unsorted_segment_prod"(%9, %10, %12) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %14 = "tfl.unsorted_segment_prod"(%9, %11, %12) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %15 = "tfl.pseudo_const"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %16 = "tfl.concatenation"(%15, %14, %13) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %17 = mhlo.dynamic_reshape %arg1, %16 : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x?x256xf32>
// CHECK-NEXT: %18 = "tfl.batch_matmul"(%8, %17) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT: %19 = mhlo.reshape %18 : (tensor<4x4x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT: return %19 : tensor<4x4x256xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.reduce
//===----------------------------------------------------------------------===//

// CHECK-LABEL: argmax
func.func @argmax(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
  %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
  %4:2 = "mhlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = mhlo.or %7, %8 : tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = mhlo.and %11, %12 : tensor<i1>
    %14 = mhlo.or %9, %13 : tensor<i1>
    %15 = "mhlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x32x256xf32>, tensor<4x32x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x32xf32>, tensor<4x32xi32>)
  func.return %4#0, %4#1 : tensor<4x32xf32>, tensor<4x32xi32>
}

// CHECK:     %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG: %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
// CHECK:     %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
// CHECK:     %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:     %4 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xf32>
// CHECK:     %5 = "tfl.arg_max"(%arg0, %cst) : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xi32>
// CHECK:     return %4, %5 : tensor<4x32xf32>, tensor<4x32xi32>

// -----

// CHECK-LABEL: argmax_constant
func.func @argmax_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %3 = mhlo.constant dense<[[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
  %4:2 = "mhlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = mhlo.or %7, %8 : tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = mhlo.and %11, %12 : tensor<i1>
    %14 = mhlo.or %9, %13 : tensor<i1>
    %15 = "mhlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x2x4xf32>, tensor<2x2x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %4#0, %4#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// CHECK-DAG: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG: %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = mhlo.constant dense<{{\[\[}}[0, 1, 2, 3], [0, 1, 2, 3]], {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
// CHECK:     %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK:     %4 = "tfl.arg_max"(%arg0, %cst) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK:     return %3, %4 : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: argmax_constant_non_z_axis
func.func @argmax_constant_non_z_axis(%arg0: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %3 = mhlo.constant dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]> : tensor<4x4xi32>
  %4:2 = "mhlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = mhlo.or %7, %8 : tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = mhlo.and %11, %12 : tensor<i1>
    %14 = mhlo.or %9, %13 : tensor<i1>
    %15 = "mhlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
  func.return %4#0, %4#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK-DAG: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG: %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = mhlo.constant dense<{{\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]> : tensor<4x4xi32>
// CHECK:     %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK:     %4 = "tfl.arg_max"(%arg0, %cst) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK:     return %3, %4 : tensor<4xf32>, tensor<4xi32>

// -----

// CHECK-LABEL: argmax_bool
func.func @argmax_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %1 = mhlo.constant dense<false> : tensor<i1>
  %2 = mhlo.constant dense<0> : tensor<i32>
  %3:2 = mhlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<2xi1>, tensor<2xi32>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i32>)
    reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %4 = mhlo.compare  GT, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = mhlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i1>
    %6 = mhlo.compare  EQ, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = mhlo.compare  LT, %arg2, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = mhlo.and %6, %7 : tensor<i1>
    %9 = mhlo.or %4, %8 : tensor<i1>
    %10 = mhlo.select %9, %arg2, %arg4 : tensor<i1>, tensor<i32>
    mhlo.return %5, %10 : tensor<i1>, tensor<i32>
  }
  return %3#1 : tensor<i32>
}

// CHECK:     %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
// CHECK-DAG: %1 = mhlo.constant dense<false> : tensor<i1>
// CHECK:     %2 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_any"(%arg0, %cst) <{keep_dims = false}> : (tensor<2xi1>, tensor<1xi32>) -> tensor<i1>
// CHECK:     %4 = "tfl.arg_max"(%arg0, %cst) : (tensor<2xi1>, tensor<1xi32>) -> tensor<i32>
// CHECK:     return %4 : tensor<i32>

// -----

// CHECK-LABEL: argmin
func.func @argmin(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
  %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
  %4:2 = "mhlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = mhlo.or %7, %8 : tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = mhlo.and %11, %12 : tensor<i1>
    %14 = mhlo.or %9, %13 : tensor<i1>
    %15 = "mhlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x32x256xf32>, tensor<4x32x256xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x32xf32>, tensor<4x32xi32>)
  func.return %4#0, %4#1 : tensor<4x32xf32>, tensor<4x32xi32>
}

// CHECK-DAG: %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:     %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
// CHECK:     %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
// CHECK:     %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:     %4 = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = false}> : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xf32>
// CHECK:     %5 = "tfl.arg_min"(%arg0, %cst) : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xi32>
// CHECK:     return %4, %5 : tensor<4x32xf32>, tensor<4x32xi32>

// -----

// CHECK-LABEL: argmin_i16
func.func @argmin_i16(%arg0: tensor<2xi16>) -> (tensor<i16>, tensor<i32>) {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %2 = mhlo.constant dense<32767> : tensor<i16>
  %3 = mhlo.constant dense<0> : tensor<i32>
  %4:2 = "mhlo.reduce"(%arg0, %1, %2, %3) ({
  ^bb0(%arg1: tensor<i16>, %arg2: tensor<i32>, %arg3: tensor<i16>, %arg4: tensor<i32>):
    %11 = mhlo.constant dense<false> : tensor<i1>
    %12 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %13 = "mhlo.select"(%12, %arg1, %arg3) : (tensor<i1>, tensor<i16>, tensor<i16>) -> tensor<i16>
    %14 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %15 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %16 = mhlo.and %14, %15 : tensor<i1>
    %17 = mhlo.or %12, %16 : tensor<i1>
    %18 = "mhlo.select"(%17, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%13, %18) : (tensor<i16>, tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi16>, tensor<2xi32>, tensor<i16>, tensor<i32>) -> (tensor<i16>, tensor<i32>)
  func.return %4#0, %4#1 : tensor<i16>, tensor<i32>
}

// CHECK:     %0 = mhlo.constant dense<false> : tensor<i1>
// CHECK:     %1 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
// CHECK-DAG: %2 = mhlo.constant dense<32767> : tensor<i16>
// CHECK:     %3 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %4 = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = false}> : (tensor<2xi16>, tensor<1xi32>) -> tensor<i16>
// CHECK:     %5 = "tfl.arg_min"(%arg0, %cst) : (tensor<2xi16>, tensor<1xi32>) -> tensor<i32>
// CHECK:     return %4, %5 : tensor<i16>, tensor<i32>

// -----

// CHECK-LABEL: argmin_constant
func.func @argmin_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %3 = mhlo.constant dense<[[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
  %4:2 = "mhlo.reduce"(%arg0, %3, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = mhlo.or %7, %8 : tensor<i1>
    %10 = "mhlo.select"(%9, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %13 = mhlo.and %11, %12 : tensor<i1>
    %14 = mhlo.or %9, %13 : tensor<i1>
    %15 = "mhlo.select"(%14, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%10, %15) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<2x2x4xf32>, tensor<2x2x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %4#0, %4#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// CHECK-DAG: %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-DAG: %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = mhlo.constant dense<{{\[\[}}[0, 1, 2, 3], [0, 1, 2, 3]], {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
// CHECK:     %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK:     %4 = "tfl.arg_min"(%arg0, %cst) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK:     return %3, %4 : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: argmin_bool
func.func @argmin_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
  %1 = mhlo.constant dense<false> : tensor<i1>
  %2 = mhlo.constant dense<0> : tensor<i32>
  %3:2 = mhlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<2xi1>, tensor<2xi32>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i32>)
    reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %4 = mhlo.compare  LT, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = mhlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<i1>
    %6 = mhlo.compare  EQ, %arg1, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = mhlo.compare  LT, %arg2, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = mhlo.and %6, %7 : tensor<i1>
    %9 = mhlo.or %4, %8 : tensor<i1>
    %10 = mhlo.select %9, %arg2, %arg4 : tensor<i1>, tensor<i32>
    mhlo.return %5, %10 : tensor<i1>, tensor<i32>
  }
  return %3#1 : tensor<i32>
}

// CHECK:     %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xi32>
// CHECK-DAG: %1 = mhlo.constant dense<false> : tensor<i1>
// CHECK:     %2 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_all"(%arg0, %cst) <{keep_dims = false}> : (tensor<2xi1>, tensor<1xi32>) -> tensor<i1>
// CHECK:     %4 = "tfl.arg_min"(%arg0, %cst) : (tensor<2xi1>, tensor<1xi32>) -> tensor<i32>
// CHECK:     return %4 : tensor<i32>

// -----

// CHECK-LABEL: argmax_with_reshaped_iota
func.func @argmax_with_reshaped_iota(%arg0: tensor<1x32x1xf32>) -> (tensor<1x1xf32>, tensor<1x1xi32>) {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<32xi32>
  %3 = "mhlo.reshape"(%2) : (tensor<32xi32>) -> tensor<1x32x1xi32>
  %4:2 = mhlo.reduce(%arg0 init: %0), (%3 init: %1) across dimensions = [1] : (tensor<1x32x1xf32>, tensor<1x32x1xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x1xf32>, tensor<1x1xi32>)
   reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %5 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = mhlo.or %5, %6 : tensor<i1>
    %8 = "mhlo.select"(%7, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = "mhlo.compare"(%arg2, %arg4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = mhlo.and %9, %10 : tensor<i1>
    %12 = mhlo.or %7, %11 : tensor<i1>
    %13 = "mhlo.select"(%12, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%8, %13) : (tensor<f32>, tensor<i32>) -> ()
  }
  func.return %4#0, %4#1 : tensor<1x1xf32>, tensor<1x1xi32>
}

// CHECK-DAG: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:     %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<32xi32>
// CHECK:     %3 = mhlo.reshape %2 : (tensor<32xi32>) -> tensor<1x32x1xi32>
// CHECK:     %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %4 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xf32>
// CHECK:     %5 = "tfl.arg_max"(%arg0, %cst) : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xi32>
// CHECK:     return %4, %5 : tensor<1x1xf32>, tensor<1x1xi32>

// -----

// CHECK-LABEL: pytorch_argmax
func.func @pytorch_argmax(%arg0: tensor<1x9xi32>) -> tensor<1xi32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<-2147483648> : tensor<i32>
  %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<9xi32>
  %3 = mhlo.reshape %2 : (tensor<9xi32>) -> tensor<1x9xi32>
  %4:2 = mhlo.reduce(%arg0 init: %1), (%3 init: %0) across dimensions = [1] : (tensor<1x9xi32>, tensor<1x9xi32>, tensor<i32>, tensor<i32>) -> (tensor<1xi32>, tensor<1xi32>)
    reducer(%arg1: tensor<i32>, %arg3: tensor<i32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
    %6 = mhlo.compare  GE, %arg1, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = mhlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<i32>
    %8 = mhlo.compare  EQ, %arg1, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = mhlo.minimum %arg2, %arg4 : tensor<i32>
    %10 = mhlo.select %6, %arg2, %arg4 : tensor<i1>, tensor<i32>
    %11 = mhlo.select %8, %9, %10 : tensor<i1>, tensor<i32>
    mhlo.return %7, %11 : tensor<i32>, tensor<i32>
  }
  func.return %4#1 : tensor<1xi32>
}

// CHECK:     %0 = mhlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %1 = mhlo.constant dense<-2147483648> : tensor<i32>
// CHECK:     %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<9xi32>
// CHECK:     %3 = mhlo.reshape %2 : (tensor<9xi32>) -> tensor<1x9xi32>
// CHECK:     %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %4 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<1x9xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:     %5 = "tfl.arg_max"(%arg0, %cst) : (tensor<1x9xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:     return %5 : tensor<1xi32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.cbrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: cbrt_f32
func.func @cbrt_f32(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> {
  %0 = "mhlo.cbrt"(%arg0) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  func.return %0 : tensor<1x32x1xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %cst_0 = arith.constant dense<3.000000e+00> : tensor<f32>
// CHECK: %0 = tfl.div %cst, %cst_0 {fused_activation_function = "NONE"} : tensor<f32>
// CHECK: %1 = tfl.pow(%arg0, %0) : (tensor<1x32x1xf32>, tensor<f32>) -> tensor<1x32x1xf32>
// CHECK: return %1 : tensor<1x32x1xf32>

// -----

// CHECK-LABEL: cbrt_f64
func.func @cbrt_f64(%arg0: tensor<1x32x1xf64>) -> tensor<1x32x1xf64> {
  %0 = "mhlo.cbrt"(%arg0) : (tensor<1x32x1xf64>) -> tensor<1x32x1xf64>
  func.return %0 : tensor<1x32x1xf64>
}

// CHECK-NOT: tfl

// -----

//===----------------------------------------------------------------------===//
// mhlo.convolution
//===----------------------------------------------------------------------===//

//
// 2D
//=---

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc
func.func @conv2d_nhwc_ohwi_nhwc(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x3xf32>, tensor<2x1x1x3xf32>, tensor<2xf32>) -> tensor<1x256x256x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_strides
func.func @conv2d_nhwc_ohwi_nhwc_no_strides(%arg0: tensor<1x64x64x4xf32>, %arg1: tensor<320x3x3x4xf32>) -> tensor<1x62x62x320xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x64x64x4xf32>, tensor<320x3x3x4xf32>) -> tensor<1x62x62x320xf32>
  func.return %0 : tensor<1x62x62x320xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x64x64x4xf32>, tensor<320x3x3x4xf32>, tensor<320xf32>) -> tensor<1x62x62x320xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_padding
func.func @conv2d_nhwc_ohwi_nhwc_no_padding(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_lhs_dilation
func.func @conv2d_nhwc_ohwi_nhwc_no_lhs_dilation(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_rhs_dilation
func.func @conv2d_nhwc_ohwi_nhwc_no_rhs_dilation(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    lhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_no_strides
func.func @conv2d_nhwc_ohwi_nhwc_no_strides(%arg0: tensor<1x6x6x207xf32>, %arg1: tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>
  } : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>) -> tensor<1x4x4x16xf32>
  func.return %0 : tensor<1x4x4x16xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x207xf32>, tensor<16x3x3x207xf32>, tensor<16xf32>) -> tensor<1x4x4x16xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_rhs_dilated
func.func @conv2d_nhwc_ohwi_nhwc_rhs_dilated(%input: tensor<1x256x256x3xf32>, %filter: tensor<2x5x5x3xf32>) -> tensor<1x248x248x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 2]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x256x256x3xf32>, tensor<2x5x5x3xf32>) -> tensor<1x248x248x2xf32>
  func.return %0 : tensor<1x248x248x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x3xf32>, tensor<2x5x5x3xf32>, tensor<2xf32>) -> tensor<1x248x248x2xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_with_strides
func.func @conv2d_nhwc_ohwi_nhwc_with_strides(%input: tensor<1x115x115x96xf32>, %filter: tensor<96x7x7x96xf32>) -> tensor<1x55x55x96xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<2> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x115x115x96xf32>, tensor<96x7x7x96xf32>) -> tensor<1x55x55x96xf32>
  func.return %0 : tensor<1x55x55x96xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x115x115x96xf32>, tensor<96x7x7x96xf32>, tensor<96xf32>) -> tensor<1x55x55x96xf32>

// -----

// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc_grouped_features
func.func @conv2d_nhwc_ohwi_nhwc_grouped_features(%input: tensor<1x256x256x6xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 2 : i64,
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x256x256x6xf32>, tensor<2x1x1x3xf32>) -> tensor<1x256x256x2xf32>
  func.return %0 : tensor<1x256x256x2xf32>
}

// CHECK: "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x6xf32>, tensor<2x1x1x3xf32>, tensor<2xf32>) -> tensor<1x256x256x2xf32>

// -----

// TODO: b/351437662 - Add support for dynamic batch.
// CHECK-LABEL: conv2d_nhwc_ohwi_nhwc
func.func @conv2d_nhwc_ohwi_nhwc_dynamic_batch(%input: tensor<?x256x256x3xf32>, %filter: tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32> {
  %0 = mhlo.convolution(%input, %filter)
    dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64
  } : (tensor<?x256x256x3xf32>, tensor<2x1x1x3xf32>) -> tensor<?x256x256x2xf32>
  func.return %0 : tensor<?x256x256x2xf32>
}

// CHECK-NOT: tfl

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc
func.func @depthwise_conv2d_nhwc_ihwo_nhwc(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x8x8x207xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x8x8x207xf32>
  func.return %0 : tensor<1x8x8x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x8x8x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_strided
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_strided(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x4x4x207xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x4x4x207xf32>
  func.return %0 : tensor<1x4x4x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x4x4x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_dilated
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_dilated(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x207xf32>) -> tensor<1x6x6x207xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<2> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>) -> tensor<1x6x6x207xf32>
  func.return %0 : tensor<1x6x6x207xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x207xf32>, tensor<207xf32>) -> tensor<1x6x6x207xf32>
// CHECK: return %0

// -----

// CHECK-LABEL: depthwise_conv2d_nhwc_ihwo_nhwc_non_trivial_depth_multiplier
func.func @depthwise_conv2d_nhwc_ihwo_nhwc_non_trivial_depth_multiplier(%arg0: tensor<1x10x10x207xf32>, %arg1: tensor<1x3x3x3519xf32>) -> tensor<1x6x6x3519xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]>,
    feature_group_count = 207 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<0> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<2> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x10x10x207xf32>, tensor<1x3x3x3519xf32>) -> tensor<1x6x6x3519xf32>
  func.return %0 : tensor<1x6x6x3519xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 17 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x10x10x207xf32>, tensor<1x3x3x3519xf32>, tensor<3519xf32>) -> tensor<1x6x6x3519xf32>
// CHECK: return %0

// -----

// TODO: b/351437662 - Add support for conv to resize.
// CHECK-LABEL: conv2d_resize_perferred_nhwc_hwoi_nhwc
func.func @conv2d_resize_perferred_nhwc_hwoi_nhwc(%arg0: tensor<1x56x1248x16xf32>, %arg1: tensor<16x3x1x1xf32>) -> tensor<1x111x1248x16xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 16 : i64,
    lhs_dilation = dense<[2, 1]> : tensor<2xi64>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    window_strides = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x56x1248x16xf32>, tensor<16x3x1x1xf32>) -> tensor<1x111x1248x16xf32>
  func.return %0 : tensor<1x111x1248x16xf32>
}

// CHECK-NOT: tfl

// -----

// TODO: b/351437662 - Add support for conv to resize.
// CHECK-LABEL: conv2d_to_resize_nhwc_hwoi_nhwc
func.func @conv2d_to_resize_nhwc_hwoi_nhwc(%arg0: tensor<1x56x624x16xf32>, %arg1: tensor<16x1x257x1xf32>) -> tensor<1x56x904x16xf32> {
	%0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>,
    feature_group_count = 16 : i64,
    lhs_dilation = dense<[1, 129]> : tensor<2xi64>,
    padding = dense<[[0, 0], [128, 128]]> : tensor<2x2xi64>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<[1, 89]> : tensor<2xi64>
  } : (tensor<1x56x624x16xf32>, tensor<16x1x257x1xf32>) -> tensor<1x56x904x16xf32>
  func.return %0 : tensor<1x56x904x16xf32>
}

// CHECK-NOT: tfl

// -----

//
// 3D
//=---

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc
func.func @conv3d_ndhwc_dhwio_ndhwc(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_strides
func.func @conv3d_ndhwc_dhwio_ndhwc_no_strides(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_pad
func.func @conv3d_ndhwc_dhwio_ndhwc_no_pad(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_lhs_dilation
func.func @conv3d_ndhwc_dhwio_ndhwc_no_lhs_dilation(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_no_rhs_dilation
func.func @conv3d_ndhwc_dhwio_ndhwc_no_rhs_dilation(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x256x256x256x2xf32>
  func.return %0 : tensor<1x256x256x256x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x256x256x256x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_rhs_dilated
func.func @conv3d_ndhwc_dhwio_ndhwc_rhs_dilated(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<5x5x5x3x2xf32>) -> tensor<1x248x248x248x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<2> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<5x5x5x3x2xf32>) -> tensor<1x248x248x248x2xf32>
  func.return %0 : tensor<1x248x248x248x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 2 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<5x5x5x3x2xf32>, tensor<2xf32>) -> tensor<1x248x248x248x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_strided
func.func @conv3d_ndhwc_dhwio_ndhwc_strided(%input: tensor<1x256x256x256x3xf32>, %filter: tensor<1x1x1x3x2xf32>) -> tensor<1x52x52x52x2xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    window_strides = dense<5> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>) -> tensor<1x52x52x52x2xf32>
  func.return %0 : tensor<1x52x52x52x2xf32>
}

// CHECK: "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 5 : i32, stride_h = 5 : i32, stride_w = 5 : i32}> : (tensor<1x256x256x256x3xf32>, tensor<1x1x1x3x2xf32>, tensor<2xf32>) -> tensor<1x52x52x52x2xf32>

// -----

// CHECK-LABEL: conv3d_ndhwc_dhwio_ndhwc_grouped_features
func.func @conv3d_ndhwc_dhwio_ndhwc_grouped_features(%input: tensor<1x256x256x256x6xf32>, %filter: tensor<1x1x1x2x6xf32>) -> tensor<1x256x256x256x6xf32> {
  %0 = "mhlo.convolution"(%input, %filter) {
    dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>,
    batch_group_count = 1 : i64,
    feature_group_count = 3 : i64,
    window_strides = dense<1> : tensor<3xi64>,
    padding = dense<0> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>
  } : (tensor<1x256x256x256x6xf32>, tensor<1x1x1x2x6xf32>) -> tensor<1x256x256x256x6xf32>
  func.return %0 : tensor<1x256x256x256x6xf32>
}


// Unlike TF, tfl.conv_3d requires kernel_dims[3] == input_dims[4]

// CHECK-NOT: tfl
// CHECK: mhlo.convolution

// -----

//===----------------------------------------------------------------------===//
// mhlo.pad
//===----------------------------------------------------------------------===//

func.func @pad_cst_zero_vals(%arg0: tensor<8x128xf32>) -> tensor<11x131xf32> {
  %pad_val = mhlo.constant dense<0.0> : tensor<f32>
  %0 = "mhlo.pad"(%arg0, %pad_val) {
    edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
  func.return %0 : tensor<11x131xf32>
}

// CHECK:      %[[PADDINGS:.*]] = arith.constant
// CHECK-SAME: [1, 2], [0, 3]
// CHECK-SAME: tensor<2x2xi64>
// CHECK:      "tfl.pad"(%arg0, %[[PADDINGS]]) : (tensor<8x128xf32>, tensor<2x2xi64>) -> tensor<11x131xf32>

// -----

func.func @pad(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
  func.return %0 : tensor<11x131xf32>
}

// CHECK:      %[[PADDINGS:.*]] = arith.constant
// CHECK-SAME: [1, 2], [0, 3]
// CHECK-SAME: tensor<2x2xi64>
// CHECK:      "tfl.padv2"(%arg0, %[[PADDINGS]], %arg1) : (tensor<8x128xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<11x131xf32>

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.multiply }
//===------------------------------------------------------------------------===

// CHECK-LABEL: int_reduce_to_prod(
func.func @int_reduce_to_prod(%arg0: tensor<1x256xi32>) -> tensor<1xi32> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = mhlo.multiply %arg1, %arg2 : tensor<i32>
    "mhlo.return"(%2) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.reduce_prod"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

// CHECK-LABEL: reduce_to_prod
func.func @reduce_to_prod(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.multiply %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.reduce_prod"(%arg0, %[[CST:.*]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_prod_non_constant_init
func.func @reduce_to_prod_non_constant_init(%arg0: tensor<1x256xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %1 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = mhlo.multiply %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %0 = "tfl.reduce_prod"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK: %1 = tfl.mul(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.add }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_sum
func.func @reduce_to_sum(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_sum_non_constant_init
func.func @reduce_to_sum_non_constant_init(%arg0: tensor<1x256xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %1 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %0 = "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK: %1 = tfl.add(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: int_reduce_to_sum
func.func @int_reduce_to_sum(%arg0: tensor<1x256xi32>) -> tensor<1xi32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = mhlo.add %arg1, %arg2 : tensor<i32>
    "mhlo.return"(%2) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.sum"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.max }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_max
func.func @reduce_to_max(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_max_int
func.func @reduce_to_max_int(%arg0: tensor<1x4xi32>) -> tensor<1xi32> {
  // -2147483648 is MIN for INT32
  %0 = mhlo.constant dense<-2147483648> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
   reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
    %892 = mhlo.maximum %arg2, %arg3 : tensor<i32>
    "mhlo.return"(%892) : (tensor<i32>) -> ()
  }
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.reduce_max"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x4xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.min }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_min
func.func @reduce_to_min(%arg0: tensor<1x256xf32>) -> tensor<1xf32> {
  // "0x7F800000" represents INF for f32.
  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.minimum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x256xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

// CHECK-LABEL: reduce_to_min_int
func.func @reduce_to_min_int(%arg0: tensor<1x4xi32>) -> tensor<1xi32> {
  // 2147483647 is MAX for INT32
  %0 = mhlo.constant dense<2147483647> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
   reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
    %892 = mhlo.minimum %arg2, %arg3 : tensor<i32>
    "mhlo.return"(%892) : (tensor<i32>) -> ()
  }
  func.return %1 : tensor<1xi32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK: %1 = "tfl.reduce_min"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x4xi32>, tensor<1xi32>) -> tensor<1xi32>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.and }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_all
func.func @reduce_to_all(%arg0: tensor<1x2x3x4x5xi1>, %arg1: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = mhlo.constant dense<true> : tensor<i1>
  %1 = "mhlo.reduce"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
        %2 = mhlo.and %arg2, %arg3 : tensor<i1>
        "mhlo.return"(%2) : (tensor<i1>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %1: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %1 = "tfl.reduce_all"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>

// -----

// CHECK-LABEL: reduce_to_all_non_constant_init
func.func @reduce_to_all_non_constant_init(%arg0: tensor<i1>, %arg1: tensor<1x2x3x4x5xi1>, %arg2: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = "mhlo.reduce"(%arg1, %arg0) ({
    ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
        %1 = mhlo.and %arg3, %arg4 : tensor<i1>
        "mhlo.return"(%1) : (tensor<i1>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %0: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %0 = "tfl.reduce_all"(%arg1, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>
// CHECK: %1 = tfl.logical_and(%0, %arg0) : (tensor<2x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce { mhlo.or }
//===------------------------------------------------------------------------===

// CHECK-LABEL: reduce_to_any
func.func @reduce_to_any(%arg0: tensor<1x2x3x4x5xi1>, %arg1: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.reduce"(%arg0, %0) ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
        %2 = mhlo.or %arg2, %arg3 : tensor<i1>
        "mhlo.return"(%2) : (tensor<i1>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %1: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %1 = "tfl.reduce_any"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>

// -----

// CHECK-LABEL: reduce_to_any_non_constant_init
func.func @reduce_to_any_non_constant_init(%arg0: tensor<i1>, %arg1: tensor<1x2x3x4x5xi1>, %arg2: tensor<2xi64>) -> tensor<2x4x5xi1> {
  %0 = "mhlo.reduce"(%arg1, %arg0) ({
    ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
        %1 = mhlo.or %arg3, %arg4 : tensor<i1>
        "mhlo.return"(%1) : (tensor<i1>) -> ()
    }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2x3x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>
  func.return %0: tensor<2x4x5xi1>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 2]> : tensor<2xi32>
// CHECK: %0 = "tfl.reduce_any"(%arg1, %[[CST]]) <{keep_dims = false}> : (tensor<1x2x3x4x5xi1>, tensor<2xi32>) -> tensor<2x4x5xi1>
// CHECK: %1 = tfl.logical_or(%0, %arg0) : (tensor<2x4x5xi1>, tensor<i1>) -> tensor<2x4x5xi1>

// -----

//===------------------------------------------------------------------------===
// mhlo.gather
//===------------------------------------------------------------------------===

// CHECK-LABEL: gather_to_slice
func.func @gather_to_slice(%arg0: tensor<3x2944xi32>, %arg1: tensor<3x2xi32>) -> tensor<3x1504xi32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1,
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1504]> : tensor<2xi64>
  } : (tensor<3x2944xi32>, tensor<3x2xi32>) -> tensor<3x1504xi32>
  func.return %0 : tensor<3x1504xi32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[2, 1440]> : tensor<2xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK:     %[[VAL_0:.*]] = "tfl.maximum"(%arg1, %[[CST_0]]) : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<3x2xi32>
// CHECK:     %[[VAL_1:.*]] = "tfl.minimum"(%[[VAL_0]], %[[CST]]) : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<3x2xi32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<[1, 1504]> : tensor<2xi32>
// CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_2]], %[[CST_3]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_3:.*]] = "tfl.squeeze"(%[[VAL_2]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_4:.*]] = "tfl.slice"(%arg0, %[[VAL_3]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK-DAG: %[[CST_5:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_5:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_4]], %[[CST_5]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_6:.*]] = "tfl.squeeze"(%[[VAL_5]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_7:.*]] = "tfl.slice"(%arg0, %[[VAL_6]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK-DAG: %[[CST_6:.*]] = arith.constant dense<[2, 0]> : tensor<2xi32>
// CHECK-DAG: %[[CST_7:.*]] = arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK:     %[[VAL_8:.*]] = "tfl.slice"(%[[VAL_1]], %[[CST_6]], %[[CST_7]]) : (tensor<3x2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_9:.*]] = "tfl.squeeze"(%[[VAL_8]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_10:.*]] = "tfl.slice"(%arg0, %[[VAL_9]], %[[CST_1]]) : (tensor<3x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>
// CHECK:     %[[VAL_11:.*]] = "tfl.concatenation"(%[[VAL_4]], %[[VAL_7]], %[[VAL_10]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x1504xi32>, tensor<1x1504xi32>, tensor<1x1504xi32>) -> tensor<3x1504xi32>

// -----

// CHECK-LABEL: gather
func.func @gather(%arg0: tensor<147456xf16>, %arg1: tensor<192x256x1xi32>) -> tensor<192x256xf16> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<147456xf16>, tensor<192x256x1xi32>) -> tensor<192x256xf16>
  func.return %0 : tensor<192x256xf16>
}

// -----

// CHECK-LABEL: gather_with_ui32indices
func.func @gather_with_ui32indices(%arg0: tensor<147456xf16>, %arg1: tensor<192x256x1xui32>) -> tensor<192x256xf16> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<147456xf16>, tensor<192x256x1xui32>) -> tensor<192x256xf16>
  func.return %0 : tensor<192x256xf16>
}

// CHECK: %0 = "tfl.cast"(%arg1) : (tensor<192x256x1xui32>) -> tensor<192x256x1xi64>

// -----

// CHECK-LABEL: gather_nd
func.func @gather_nd(%arg0: tensor<98x128xf32>, %arg1: tensor<4x64xi32>) -> tensor<4x64x128xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
			offset_dims = [2],
			start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 128]> : tensor<2xi64>
  } : (tensor<98x128xf32>, tensor<4x64xi32>) -> tensor<4x64x128xf32>
  func.return %0 : tensor<4x64x128xf32>
}

// CHECK: %[[VAL_0:.*]] = mhlo.reshape %arg1 : (tensor<4x64xi32>) -> tensor<4x64x1xi32>
// CHECK: %[[VAL_1:.*]] = "tfl.gather_nd"(%arg0, %[[VAL_0]]) : (tensor<98x128xf32>, tensor<4x64x1xi32>) -> tensor<4x64x128xf32>

// -----

// CHECK-LABEL: gather_transpose
func.func @gather_transpose(%arg0: tensor<128x256xf32>, %arg1: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [1],
      index_vector_dim = 1,
			offset_dims = [1],
			start_index_map = [1],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[128, 1]> : tensor<2xi64>
  } : (tensor<128x256xf32>, tensor<4x1xi32>) -> tensor<4x128xf32>
  func.return %0 : tensor<4x128xf32>
}

// CHECK: %[[VAL_0:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK: %[[VAL_1:.*]] = "tfl.cast"(%[[VAL_0]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_2:.*]] = "tfl.transpose"(%arg0, %[[VAL_1]]) : (tensor<128x256xf32>, tensor<2xi32>) -> tensor<256x128xf32>
// CHECK: %[[VAL_3:.*]] = "tfl.gather_nd"(%[[VAL_2]], %arg1) : (tensor<256x128xf32>, tensor<4x1xi32>) -> tensor<4x128xf32>

// -----

// CHECK-LABEL: gather_offset
func.func @gather_offset(%arg0: tensor<1x20xi32>, %arg1: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [1],
      index_vector_dim = 1,
			offset_dims = [0],
			start_index_map = [1],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<1> : tensor<2xi64>
  } : (tensor<1x20xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
  func.return %0 : tensor<1x1xi32>
}

// CHECK: %[[VAL_0:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK: %[[VAL_1:.*]] = "tfl.cast"(%[[VAL_0]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_2:.*]] = "tfl.transpose"(%arg0, %[[VAL_1]]) : (tensor<1x20xi32>, tensor<2xi32>) -> tensor<20x1xi32>
// CHECK: %[[VAL_3:.*]] = "tfl.gather_nd"(%[[VAL_2]], %arg1) : (tensor<20x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
// CHECK: %[[VAL_4:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK: %[[VAL_5:.*]] = "tfl.cast"(%[[VAL_4]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_6:.*]] = "tfl.transpose"(%[[VAL_3]], %[[VAL_5]]) : (tensor<1x1xi32>, tensor<2xi32>) -> tensor<1x1xi32>

// -----

// CHECK-LABEL: gather_to_slice_batch_size_1
func.func @gather_to_slice_batch_size_1(%arg0: tensor<1x2944xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x1504xi32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1,
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1504]> : tensor<2xi64>
  } : (tensor<1x2944xi32>, tensor<1x2xi32>) -> tensor<1x1504xi32>
  func.return %0 : tensor<1x1504xi32>
}

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1440]> : tensor<2xi32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK:     %[[VAL_0:.*]] = "tfl.maximum"(%arg1, %[[CST_0]]) : (tensor<1x2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[VAL_1:.*]] = "tfl.minimum"(%[[VAL_0]], %[[CST]]) : (tensor<1x2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<[1, 1504]> : tensor<2xi32>
// CHECK:     %[[VAL_2:.*]] = "tfl.squeeze"(%[[VAL_1]]) <{squeeze_dims = [0]}> : (tensor<1x2xi32>) -> tensor<2xi32>
// CHECK:     %[[VAL_3:.*]] = "tfl.slice"(%arg0, %[[VAL_2]], %[[CST_1]]) : (tensor<1x2944xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x1504xi32>

// -----

// CHECK-LABEL: gather_slice_dynamic_indices
func.func @gather_slice_dynamic_indices(%arg0: tensor<256000x1024xi8>, %arg1: tensor<?x?x1xi32>) -> tensor<?x?x1024xi8> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = dense<[1, 1024]> : tensor<2xi64>
  } : (tensor<256000x1024xi8>, tensor<?x?x1xi32>) -> tensor<?x?x1024xi8>
  func.return %0 : tensor<?x?x1024xi8>
}

// CHECK: %0 = "tfl.gather_nd"(%arg0, %arg1) : (tensor<256000x1024xi8>, tensor<?x?x1xi32>) -> tensor<?x?x1024xi8>

// -----

// CHECK-LABEL: gather_scalar_dynamic_indices
func.func @gather_scalar_dynamic_indices(%arg0: tensor<256000xf32>, %arg1: tensor<?x?x1xi32>) -> tensor<?x?xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = dense<1> : tensor<1xi64>
  } : (tensor<256000xf32>, tensor<?x?x1xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %0 = "tfl.gather_nd"(%arg0, %arg1) : (tensor<256000xf32>, tensor<?x?x1xi32>) -> tensor<?x?xf32>

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce_window -> avg pool
//===------------------------------------------------------------------------===

// CHECK-LABEL: avgpool_same_channel_first
func.func @avgpool_same_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x8x8xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<4x16x16x3xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.transpose"(%arg0) <{permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %3 = "mhlo.reduce_window"(%2, %1) <{base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %8 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %8 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %4 = "mhlo.transpose"(%3) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  %5 = "mhlo.reduce_window"(%0, %1) <{base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %8 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %8 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %6 = "mhlo.transpose"(%5) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  %7 = mhlo.divide %4, %6 : tensor<4x3x8x8xf32>
  return %7 : tensor<4x3x8x8xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK-SAME: (tensor<4x3x16x16xf32>, tensor<4xi32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x8x8x3xf32>
// CHECK:      %[[TPOSED_OUT:.*]] = "tfl.transpose"(%[[POOL_OUT]]
// CHECK-SAME: (tensor<4x8x8x3xf32>, tensor<4xi32>) -> tensor<4x3x8x8xf32>
// CHECK:      return %[[TPOSED_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_channel_first
func.func @avgpool_valid_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x7x7xf32> {
  %0 = mhlo.constant dense<9.000000e+00> : tensor<4x3x7x7xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.transpose"(%arg0) <{permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %3 = "mhlo.reduce_window"(%2, %1) <{base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %6 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %6 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x7x7x3xf32>
  %4 = "mhlo.transpose"(%3) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<4x7x7x3xf32>) -> tensor<4x3x7x7xf32>
  %5 = mhlo.divide %4, %0 : tensor<4x3x7x7xf32>
  return %5 : tensor<4x3x7x7xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK-SAME: (tensor<4x3x16x16xf32>, tensor<4xi32>) -> tensor<4x16x16x3xf32>
// CHECK:      %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x7x7x3xf32>
// CHECK:      %[[TPOSED_OUT:.*]] = "tfl.transpose"(%[[POOL_OUT]]
// CHECK-SAME: (tensor<4x7x7x3xf32>, tensor<4xi32>) -> tensor<4x3x7x7xf32>
// CHECK:      return %[[TPOSED_OUT]]

// -----

func.func @avgpool_valid(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = mhlo.constant dense<0.0> : tensor<f32>
  %1 = mhlo.constant dense<9.0> : tensor<4x7x7x8xf32>
  %2 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %5 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%5) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = mhlo.divide %2, %1 : tensor<4x7x7x8xf32>
  func.return %3 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_broadcasted_divisor
func.func @avgpool_valid_broadcasted_divisor(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = mhlo.constant dense<0.0> : tensor<f32>
  %1 = mhlo.constant dense<9.0> : tensor<f32>
  %2 = "mhlo.broadcast_in_dim"(%1) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %5 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%5) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = mhlo.divide %3, %2 : tensor<4x7x7x8xf32>
  func.return %4 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_rw
func.func @avgpool_valid_rw(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<4x16x16x8xf32>
  %1 = mhlo.constant dense<0.0> : tensor<f32>
  %2 = "mhlo.reduce_window"(%arg0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %3 = "mhlo.reduce_window"(%0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = mhlo.divide %2, %3 : tensor<4x7x7x8xf32>
  func.return %4 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_valid_rw_broadcasted_const_lhs
func.func @avgpool_valid_rw_broadcasted_const_lhs(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4x16x16x8xf32>
  %2 = mhlo.constant dense<0.0> : tensor<f32>
  %3 = "mhlo.reduce_window"(%arg0, %2) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %4 = "mhlo.reduce_window"(%1, %2) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  %5 = mhlo.divide %3, %4 : tensor<4x7x7x8xf32>
  func.return %5 : tensor<4x7x7x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_same
func.func @avgpool_same(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<4x16x16x8xf32>
  %1 = mhlo.constant dense<0.0> : tensor<f32>
  %2 = "mhlo.reduce_window"(%arg0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %3 = "mhlo.reduce_window"(%0, %1) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %4 = mhlo.divide %2, %3 : tensor<4x8x8x8xf32>
  func.return %4 : tensor<4x8x8x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

// CHECK-LABEL: avgpool_reshape_broadcast
func.func @avgpool_reshape_broadcast(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<1x16x16x1xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce_window"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  %3 = "mhlo.reduce_window"(%0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x16x16x1xf32>, tensor<f32>) -> tensor<1x8x8x1xf32>
  %4 = mhlo.reshape %3 : (tensor<1x8x8x1xf32>) -> tensor<8x8xf32>
  %5 = "mhlo.broadcast_in_dim"(%4) <{broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}> : (tensor<8x8xf32>) -> tensor<4x8x8x8xf32>
  %6 = mhlo.divide %2, %5 : tensor<4x8x8x8xf32>
  return %6 : tensor<4x8x8x8xf32>
}

// CHECK: %[[POOL_OUT:.*]] = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>
// CHECK: return %[[POOL_OUT]]

// -----

//===------------------------------------------------------------------------===
// mhlo.reduce_window -> max pool
//===------------------------------------------------------------------------===

// CHECK-LABEL: maxpool_same
func.func @maxpool_same(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x8x8x8xf32>
  func.return %1 : tensor<4x8x8x8xf32>
}

// CHECK: %1 = "tfl.max_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x8x8x8xf32>

// -----

// CHECK-LABEL: maxpool_valid
func.func @maxpool_valid(%arg0: tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %6 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%6) : (tensor<f32>) -> ()
    }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x16x16x8xf32>, tensor<f32>) -> tensor<4x7x7x8xf32>
  func.return %1 : tensor<4x7x7x8xf32>
}

// CHECK: %1 = "tfl.max_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x8xf32>) -> tensor<4x7x7x8xf32>

// -----

// CHECK-LABEL: maxpool_valid_channel_first
func.func @maxpool_valid_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x7x7xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.transpose"(%arg0) <{permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %2 = "mhlo.reduce_window"(%1, %0) <{base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    mhlo.return %4 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x7x7x3xf32>
  %3 = "mhlo.transpose"(%2) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<4x7x7x3xf32>) -> tensor<4x3x7x7xf32>
  return %3 : tensor<4x3x7x7xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK:      "tfl.max_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x7x7x3xf32>
// CHECK:      return
// CHECK-SAME: tensor<4x3x7x7xf32>

// -----

// CHECK-LABEL: maxpool_same_channel_first
func.func @maxpool_same_channel_first(%arg0: tensor<4x3x16x16xf32>) -> tensor<4x3x8x8xf32> {
  // "0xFF800000" represents -INF for f32.
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.transpose"(%arg0) <{permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>}> : (tensor<4x3x16x16xf32>) -> tensor<4x16x16x3xf32>
  %2 = "mhlo.reduce_window"(%1, %0) <{base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    mhlo.return %4 : tensor<f32>
  }) : (tensor<4x16x16x3xf32>, tensor<f32>) -> tensor<4x8x8x3xf32>
  %3 = "mhlo.transpose"(%2) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<4x8x8x3xf32>) -> tensor<4x3x8x8xf32>
  return %3 : tensor<4x3x8x8xf32>
}

// CHECK:      %[[TPOSED_ARG0:.*]] = "tfl.transpose"(%arg0
// CHECK:      "tfl.max_pool_2d"(%[[TPOSED_ARG0]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<4x16x16x3xf32>) -> tensor<4x8x8x3xf32>
// CHECK:      return
// CHECK-SAME: tensor<4x3x8x8xf32>


// -----

//===----------------------------------------------------------------------===//
// mhlo.slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: slice
func.func @slice(%arg0: tensor<1x4672xf32>) -> tensor<1x519xf32> {
  %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 4672]> : tensor<2xi64>, start_indices = dense<[0, 4153]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<1x4672xf32>) -> tensor<1x519xf32>
  func.return %0 : tensor<1x519xf32>
}

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 4153]> : tensor<2xi64>
// CHECK: %[[CST_0:.*]] = arith.constant dense<[1, 4672]> : tensor<2xi64>
// CHECK: %[[CST_1:.*]] = arith.constant dense<1> : tensor<2xi64>
// CHECK: %[[VAL_0:.*]] = "tfl.cast"(%[[CST]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_1:.*]] = "tfl.cast"(%[[CST_0]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_2:.*]] = "tfl.cast"(%[[CST_1]]) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %[[VAL_3:.*]] = "tfl.strided_slice"(%arg0, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x4672xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x519xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.sort
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sort_to_topk_iota_broadcast
func.func @sort_to_topk_iota_broadcast(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = "mhlo.iota"() <{ iota_dimension = 0 : i64 }> : () -> tensor<6xi32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[1]> : tensor<1xi64>, name = "broadcast.0"}> : (tensor<6xi32>) -> tensor<3x6xi32>
  %2:2 = "mhlo.sort"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "mhlo.compare"(%arg1, %arg2) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %2#0, %2#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: %cst = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %cst) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)

// -----

// CHECK-LABEL: sort_to_topk_iota_cst_broadcast
func.func @sort_to_topk_iota_cst_broadcast(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = mhlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[1]> : tensor<1xi64>, name = "broadcast.0"}> : (tensor<6xi32>) -> tensor<3x6xi32>
  %2:2 = "mhlo.sort"(%arg0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "mhlo.compare"(%arg1, %arg2) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %2#0, %2#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: %cst = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %cst) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)

// -----

// CHECK-LABEL: sort_to_topk_const
func.func @sort_to_topk_const(%arg0: tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<3x6xi32>) {
  %0 = mhlo.constant dense<[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]> : tensor<3x6xi32>
  %1:2 = "mhlo.sort"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %3 = "mhlo.compare"(%arg1, %arg2) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x6xf32>, tensor<3x6xi32>) -> (tensor<3x6xf32>, tensor<3x6xi32>)
  func.return %1#0, %1#1 : tensor<3x6xf32>, tensor<3x6xi32>
}

// CHECK: %cst = arith.constant dense<6> : tensor<i32>
// CHECK: %values, %indices = "tfl.topk_v2"(%arg0, %cst) : (tensor<3x6xf32>, tensor<i32>) -> (tensor<3x6xf32>, tensor<3x6xi32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.dynamic_slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: dynamic_slice
func.func @dynamic_slice(%arg0: tensor<7x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x2xf32> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = dense<[4, 2]> : tensor<2xi64>}> : (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_IS_3:.*]] = arith.constant dense<3> : tensor<i32>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_3]], %[[MAX_1]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<[4, 2]> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi32>, tensor<2xi64>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_i64
func.func @dynamic_slice_i64(%arg0: tensor<7x3xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<4x2xf32> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = dense<[4, 2]> : tensor<2xi64>}> : (tensor<7x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[CST_IS_3:.*]] = arith.constant dense<3> : tensor<i64>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_3]], %[[MAX_1]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<[4, 2]> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: dynamic_slice_splat_sizes
func.func @dynamic_slice_splat_sizes(%arg0: tensor<7x3xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<2x2xf32> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = dense<2> : tensor<2xi64>}> : (tensor<7x3xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-DAG: %[[CST_IS_0:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST_IS_5:.*]] = arith.constant dense<5> : tensor<i32>
// CHECK:     %[[MAX_1:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_1:.*]] = "tfl.minimum"(%[[CST_IS_5]], %[[MAX_1]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[CST_IS_1:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:     %[[MAX_2:.*]] = "tfl.maximum"(%[[CST_IS_0]], %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[MIN_2:.*]] = "tfl.minimum"(%[[CST_IS_1]], %[[MAX_2]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:     %[[PACK:.*]] = "tfl.pack"(%[[MIN_1]], %[[MIN_2]]) <{axis = 0 : i32, values_count = 2 : i32}> : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %[[SLICE_SIZE:.*]] = arith.constant dense<2> : tensor<2xi64>
// CHECK:     "tfl.slice"(%arg0, %[[PACK]], %[[SLICE_SIZE]]) : (tensor<7x3xf32>, tensor<2xi32>, tensor<2xi64>) -> tensor<2x2xf32>

// -----

//===----------------------------------------------------------------------===//
// mhlo.compare
//===----------------------------------------------------------------------===//

// CHECK-LABEL: greater_unsupported_compare_type
func.func @greater_unsupported_compare_type(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK-NOT: tfl
// CHECK:     mhlo.compare

// -----

// CHECK-LABEL: equal
func.func @equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.equal

// -----

// CHECK-LABEL: notequal
func.func @notequal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.not_equal

// -----

// CHECK-LABEL: greater
func.func @greater(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.greater

// -----

// CHECK-LABEL: greater_equal
func.func @greater_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.greater_equal

// -----

// CHECK-LABEL: less
func.func @less(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.less

// -----

// CHECK-LABEL: less_equal
func.func @less_equal(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK: tfl.less_equal
