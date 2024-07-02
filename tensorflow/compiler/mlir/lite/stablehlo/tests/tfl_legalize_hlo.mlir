// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
// CHECK-LABEL: main
// CHECK: return %arg0 : tensor<5x7xf32>
}

// -----

// - transpose
//

func.func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>

// CHECK-LABEL:   transpose_2d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK-NEXT:    return %2 : tensor<3x2xf32>
}

// -----

func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[2, 1, 0]> : tensor<3xi64>}> : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>

// CHECK-LABEL:   transpose_3d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() <{value = dense<[2, 1, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:    return %2 : tensor<3x2x1xf32>
}

// -----

func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<?x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>

// CHECK-LABEL:   transpose_dynamic_2d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<4x?xf32>
// CHECK-NEXT:    return %2 : tensor<4x?xf32>
}

// -----

// - dot_general
//

// CHECK-LABEL: convert_dot_general
func.func @convert_dot_general(%arg0: tensor<3x2x6x5x1xf32>, %arg1: tensor<3x2x4x6xf32>) -> tensor<3x5x1x4xf32> {
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


// CHECK:      %[[TRANSPOSED_0:.*]] = "tfl.transpose"
// CHECK:      %[[TRANSPOSED_1:.*]] = "tfl.transpose"
// CHECK:      %[[RESHAPED_0:.*]] = "tfl.reshape"
// CHECK-SAME: %[[TRANSPOSED_0]] 
// CHECK:      %[[RESHAPED_1:.*]] = "tfl.reshape"
// CHECK-SAME: %[[TRANSPOSED_1]]
// CHECK:      %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<3x5x12xf32>, tensor<3x12x4xf32>) -> tensor<3x5x4xf32>
// CHECK:      %[[RESHAPED_BMM:.*]] = "tfl.reshape"
// CHECK-SAME: %[[BMM_0]]
// CHECK:      return %[[RESHAPED_BMM]] : tensor<3x5x1x4xf32>
}


// -----

// CHECK-LABEL: convert_dot_general_repeated
func.func @convert_dot_general_repeated(%arg0: tensor<1x1x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1x1x1024xf32> {
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


// CHECK:      %[[RESHAPED_0:.*]] = "tfl.reshape"
// CHECK-SAME: %arg0
// CHECK:      %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : {{.*}} -> tensor<1x1024xf32>
// CHECK:      %[[RESHAPED_BMM:.*]] = "tfl.reshape"
// CHECK-SAME: %[[BMM_0]]
// CHECK-NEXT: return %[[RESHAPED_BMM]] : tensor<1x1x1024xf32>
}

// -----

// CHECK-LABEL: convert_dot_general_int8
func.func @convert_dot_general_int8(%arg0: tensor<256xi8>, %arg1: tensor<256x8xi8>) -> tensor<8xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<256xi8>, tensor<256x8xi8>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>


// CHECK:      %[[RESHAPED_0:.*]] = "tfl.reshape"
// CHECK-SAME: %arg0
// CHECK:      %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : {{.*}} -> tensor<1x8xi32>
// CHECK:      %[[RESHAPED_BMM:.*]] = "tfl.reshape"
// CHECK-SAME: %[[BMM_0]]
// CHECK:      return %[[RESHAPED_BMM]] : tensor<8xi32>
}

// -----

// CHECK-LABEL:     convert_dot_general_dynamic_rhs_out_dim
func.func @convert_dot_general_dynamic_rhs_out_dim(%arg0: tensor<4x4x256xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x?xf32> {
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
// CHECK-NEXT: %3 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %4 = "tfl.pseudo_const"() <{value = dense<[-1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %5 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %6 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %9 = "tfl.pseudo_const"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %10 = "tfl.concatenation"(%9, %8, %7) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %11 = "tfl.cast"(%10) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %12 = "tfl.reshape"(%2, %11) : (tensor<4x256x?xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT: %13 = "tfl.batch_matmul"(%arg0, %12) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x4x256xf32>, tensor<4x256x?xf32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: %14 = "tfl.shape"(%arg0) : (tensor<4x4x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %15 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %17 = "tfl.gather"(%14, %16) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<3xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %18 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-NEXT: %19 = "tfl.gather"(%15, %18) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<3xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT: %20 = "tfl.concatenation"(%17, %19) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %21 = "tfl.cast"(%20) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %22 = "tfl.reshape"(%13, %21) : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: return %22 : tensor<4x4x?xf32>


// -----

// CHECK-LABEL:     convert_dot_general_dynamic_batch_dim
func.func @convert_dot_general_dynamic_batch_dim(%arg0: tensor<2x?x2x3xf32>, %arg1: tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32> {
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
// CHECK-NEXT: %12 = "tfl.cast"(%11) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT: %13 = "tfl.reshape"(%arg0, %12) : (tensor<2x?x2x3xf32>, tensor<4xi32>) -> tensor<2x?x2x3xf32>
// CHECK-NEXT: %14 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %15 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %17 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %18 = "tfl.unsorted_segment_prod"(%14, %15, %17) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %19 = "tfl.unsorted_segment_prod"(%14, %16, %17) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %20 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %21 = "tfl.gather"(%14, %20) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %22 = "tfl.concatenation"(%21, %19, %18) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT: %23 = "tfl.cast"(%22) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT: %24 = "tfl.reshape"(%2, %23) : (tensor<2x?x3x4xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT: %25 = "tfl.batch_matmul"(%13, %24) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x?x2x3xf32>, tensor<2x?x3x4xf32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT: %26 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %27 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %28 = "tfl.pseudo_const"() <{value = dense<[0, 1, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %29 = "tfl.gather"(%26, %28) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %30 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-NEXT: %31 = "tfl.gather"(%27, %30) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT: %32 = "tfl.concatenation"(%29, %31) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT: %33 = "tfl.cast"(%32) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT: %34 = "tfl.reshape"(%25, %33) : (tensor<2x?x2x4xf32>, tensor<4xi32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT: return %34 : tensor<2x?x2x4xf32>

// -----

// CHECK-LABEL: convert_dot_general_dynamic_lhs_rhs_out_dims
func.func @convert_dot_general_dynamic_lhs_rhs_out_dims(%arg0: tensor<2x2x?x3xf32>, %arg1: tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32> {
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
// CHECK-NEXT: %11 = "tfl.cast"(%10) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %12 = "tfl.reshape"(%arg0, %11) : (tensor<2x2x?x3xf32>, tensor<3xi32>) -> tensor<2x?x3xf32>
// CHECK-NEXT: %13 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %14 = "tfl.pseudo_const"() <{value = dense<[-1, 0, 0, -1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %15 = "tfl.pseudo_const"() <{value = dense<[-1, -1, -1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %17 = "tfl.unsorted_segment_prod"(%13, %14, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %18 = "tfl.unsorted_segment_prod"(%13, %15, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %19 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %20 = "tfl.concatenation"(%19, %18, %17) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %21 = "tfl.cast"(%20) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %22 = "tfl.reshape"(%2, %21) : (tensor<2x3x4x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK-NEXT: %23 = "tfl.batch_matmul"(%12, %22) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x?x3xf32>, tensor<2x3x?xf32>) -> tensor<2x?x?xf32>
// CHECK-NEXT: %24 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %25 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT: %26 = "tfl.pseudo_const"() <{value = dense<[0, 1, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-NEXT: %27 = "tfl.gather"(%24, %26) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT: %28 = "tfl.pseudo_const"() <{value = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-NEXT: %29 = "tfl.gather"(%25, %28) <{axis = 0 : i32, batch_dims = 0 : i32}> : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT: %30 = "tfl.concatenation"(%27, %29) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<2xi32>) -> tensor<5xi32>
// CHECK-NEXT: %31 = "tfl.cast"(%30) : (tensor<5xi32>) -> tensor<5xi32>
// CHECK-NEXT: %32 = "tfl.reshape"(%23, %31) : (tensor<2x?x?xf32>, tensor<5xi32>) -> tensor<2x2x?x4x?xf32>
// CHECK-NEXT: return %32 : tensor<2x2x?x4x?xf32>

// -----

// CHECK-LABEL: convert_dot_general_dynamic_contracting_dim
func.func @convert_dot_general_dynamic_contracting_dim(%arg0: tensor<4x4x?xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x256xf32> {
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
// CHECK-NEXT: %8 = "tfl.cast"(%7) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %9 = "tfl.reshape"(%arg0, %8) : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT: %10 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT: %11 = "tfl.pseudo_const"() <{value = dense<[-1, -1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %12 = "tfl.pseudo_const"() <{value = dense<[-1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT: %13 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %14 = "tfl.unsorted_segment_prod"(%10, %11, %13) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %15 = "tfl.unsorted_segment_prod"(%10, %12, %13) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT: %16 = "tfl.pseudo_const"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT: %17 = "tfl.concatenation"(%16, %15, %14) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT: %18 = "tfl.cast"(%17) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT: %19 = "tfl.reshape"(%arg1, %18) : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x?x256xf32>
// CHECK-NEXT: %20 = "tfl.batch_matmul"(%9, %19) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT: return %20 : tensor<4x4x256xf32>

// -----

// CHECK-LABEL: convert_argmax
func.func @convert_argmax(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
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

// CHECK-LABEL: convert_argmax_constant
func.func @convert_argmax_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
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

// CHECK-DAG:  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:  %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:      %2 = mhlo.constant dense<{{\[\[}}[0, 1, 2, 3], [0, 1, 2, 3]], {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
// CHECK:      %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:      %3 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK:      %4 = "tfl.arg_max"(%arg0, %cst) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK:      return %3, %4 : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: convert_argmax_constant_non_z_axis
func.func @convert_argmax_constant_non_z_axis(%arg0: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
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

// CHECK-DAG:  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:  %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:      %2 = mhlo.constant dense<{{\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]> : tensor<4x4xi32>
// CHECK:      %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:      %3 = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = false}> : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK:      %4 = "tfl.arg_max"(%arg0, %cst) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK:      return %3, %4 : tensor<4xf32>, tensor<4xi32>

// -----

// CHECK-LABEL: convert_argmax_bool
func.func @convert_argmax_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
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

// CHECK-LABEL: convert_argmin
func.func @convert_argmin(%arg0: tensor<4x32x256xf32>) -> (tensor<4x32xf32>, tensor<4x32xi32>) {
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

// CHECK-DAG:  %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK:      %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK:      %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<256xi32>
// CHECK:      %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<256xi32>) -> tensor<4x32x256xi32>
// CHECK:      %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:      %4 = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = false}> : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xf32>
// CHECK:      %5 = "tfl.arg_min"(%arg0, %cst) : (tensor<4x32x256xf32>, tensor<1xi32>) -> tensor<4x32xi32>
// CHECK:      return %4, %5 : tensor<4x32xf32>, tensor<4x32xi32>

// -----

// CHECK-LABEL: convert_argmin_i16
func.func @convert_argmin_i16(%arg0: tensor<2xi16>) -> (tensor<i16>, tensor<i32>) {
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

// CHECK-LABEL: convert_argmin_constant
func.func @convert_argmin_constant(%arg0: tensor<2x2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
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
// CHECK-DAG: %2 = mhlo.constant dense<{{\[\[}}[0, 1, 2, 3], [0, 1, 2, 3]], {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x2x4xi32>
// CHECK-DAG: %cst = arith.constant dense<2> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = false}> : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xf32>
// CHECK:     %4 = "tfl.arg_min"(%arg0, %cst) : (tensor<2x2x4xf32>, tensor<1xi32>) -> tensor<2x2xi32>
// CHECK:     return %3, %4 : tensor<2x2xf32>, tensor<2x2xi32>

// -----

// CHECK-LABEL: convert_argmin_bool
func.func @convert_argmin_bool(%arg0: tensor<2xi1>) -> tensor<i32> {
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
// CHECK-DAG: %2 = mhlo.constant dense<0> : tensor<i32>
// CHECK:     %cst = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %3 = "tfl.reduce_all"(%arg0, %cst) <{keep_dims = false}> : (tensor<2xi1>, tensor<1xi32>) -> tensor<i1>
// CHECK:     %4 = "tfl.arg_min"(%arg0, %cst) : (tensor<2xi1>, tensor<1xi32>) -> tensor<i32>
// CHECK:     return %4 : tensor<i32>

// -----

// CHECK-LABEL: convert_argmax_with_reshaped_iota
func.func @convert_argmax_with_reshaped_iota(%arg0: tensor<1x32x1xf32>) -> (tensor<1x1xf32>, tensor<1x1xi32>) {
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
// CHECK-DAG: %1 = mhlo.constant dense<0> : tensor<i32>
// CHECK: %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<32xi32>
// CHECK: %cst = arith.constant dense<[1, 32, 1]> : tensor<3xi64>
// CHECK: %3 = "tfl.cast"(%cst) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK: %4 = "tfl.reshape"(%2, %3) : (tensor<32xi32>, tensor<3xi32>) -> tensor<1x32x1xi32>
// CHECK: %cst_0 = arith.constant dense<1> : tensor<1xi32>
// CHECK: %5 = "tfl.reduce_max"(%arg0, %cst_0) <{keep_dims = false}> : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xf32>
// CHECK: %6 = "tfl.arg_max"(%arg0, %cst_0) : (tensor<1x32x1xf32>, tensor<1xi32>) -> tensor<1x1xi32>
// CHECK: return %5, %6 : tensor<1x1xf32>, tensor<1x1xi32>

// -----

// CHECK-LABEL: convert_pytorch_argmax
func.func @convert_pytorch_argmax(%arg0: tensor<1x9xi32>) -> tensor<1xi32> {
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

// CHECK-DAG: %0 = mhlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %1 = mhlo.constant dense<-2147483648> : tensor<i32>
// CHECK:     %2 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<9xi32>
// CHECK:     %cst = arith.constant dense<[1, 9]> : tensor<2xi64>
// CHECK:     %3 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK:     %4 = "tfl.reshape"(%2, %3) : (tensor<9xi32>, tensor<2xi32>) -> tensor<1x9xi32>
// CHECK:     %cst_0 = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %5 = "tfl.reduce_max"(%arg0, %cst_0) <{keep_dims = false}> : (tensor<1x9xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:     %6 = "tfl.arg_max"(%arg0, %cst_0) : (tensor<1x9xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:     return %6 : tensor<1xi32>

// -----

// CHECK-LABEL: lower_cbrt_f32
func.func @lower_cbrt_f32(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> {
  %0 = "mhlo.cbrt"(%arg0) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  func.return %0 : tensor<1x32x1xf32>
}

// CHECK-DAG: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %cst_0 = arith.constant dense<3.000000e+00> : tensor<f32>
// CHECK:     %0 = tfl.div %cst, %cst_0 {fused_activation_function = "NONE"} : tensor<f32>
// CHECK:     %1 = tfl.pow(%arg0, %0) : (tensor<1x32x1xf32>, tensor<f32>) -> tensor<1x32x1xf32>
// CHECK:     return %1 : tensor<1x32x1xf32>

// -----

// CHECK-LABEL: no_lower_cbrt_f64
func.func @no_lower_cbrt_f64(%arg0: tensor<1x32x1xf64>) -> tensor<1x32x1xf64> {
  %0 = "mhlo.cbrt"(%arg0) : (tensor<1x32x1xf64>) -> tensor<1x32x1xf64>
  func.return %0 : tensor<1x32x1xf64>
}

// CHECK-NOT: tfl

// -----

// CHECK-LABEL: legalize_reshape
func.func @legalize_reshape(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// CHECK: %cst = arith.constant dense<[3, 2]> : tensor<2xi64>
// CHECK: %0 = "tfl.cast"(%cst) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>

// -----

// CHECK-LABEL: legalize_dynamic_reshape_i32
func.func @legalize_dynamic_reshape_i32(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %0 = "tfl.cast"(%arg1) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

// CHECK-LABEL: legalize_dynamic_reshape_i64
func.func @legalize_dynamic_reshape_i64(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK: %0 = "tfl.cast"(%arg1) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK: %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>






