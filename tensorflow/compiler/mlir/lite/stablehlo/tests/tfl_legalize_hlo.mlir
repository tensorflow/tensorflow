// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
// CHECK-LABEL: main
// CHECK: return %arg0 : tensor<5x7xf32>
}

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

// CHECK-LABEL:   convert_dot_general
// CHECK:         %[[TRANSPOSED_0:.*]] = "mhlo.transpose"(%arg0)
// CHECK-NEXT:    %[[TRANSPOSED_1:.*]] = "mhlo.transpose"(%arg1)
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = mhlo.reshape %[[TRANSPOSED_0]]
// CHECK-NEXT:    %[[RESHAPED_1:.*]] = mhlo.reshape %[[TRANSPOSED_1]]
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<3x5x12xf32>, tensor<3x12x4xf32>) -> tensor<3x5x4xf32>
// CHECK-NEXT:    %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT:    return %[[RESHAPED_BMM]] : tensor<3x5x1x4xf32>
}


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

// CHECK-LABEL:   convert_dot_general_repeated
// CHECK:         %[[TRANSPOSED_0:.*]] = "mhlo.transpose"(%arg0)
// CHECK-NEXT:    %[[TRANSPOSED_1:.*]] = "mhlo.transpose"(%arg1)
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = mhlo.reshape %[[TRANSPOSED_0]]
// CHECK-NEXT:    %[[RESHAPED_1:.*]] = mhlo.reshape %[[TRANSPOSED_1]]
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : {{.*}} -> tensor<1x1024xf32>
// CHECK-NEXT:    %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT:    return %[[RESHAPED_BMM]] : tensor<1x1x1024xf32>
}

func.func @convert_dot_general_int8(%arg0: tensor<256xi8>, %arg1: tensor<256x8xi8>) -> tensor<8xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]>,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<256xi8>, tensor<256x8xi8>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>

// CHECK-LABEL:   convert_dot_general_int8
// CHECK:         %[[TRANSPOSED_0:.*]] = "mhlo.transpose"(%arg0)
// CHECK-NEXT:    %[[TRANSPOSED_1:.*]] = "mhlo.transpose"(%arg1)
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = mhlo.reshape %[[TRANSPOSED_0]]
// CHECK-NEXT:    %[[RESHAPED_1:.*]] = mhlo.reshape %[[TRANSPOSED_1]]
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : {{.*}} -> tensor<1x8xi32>
// CHECK-NEXT:    %[[RESHAPED_BMM:.*]] = mhlo.reshape %[[BMM_0]]
// CHECK-NEXT:    return %[[RESHAPED_BMM]] : tensor<8xi32>
}

func.func @convert_dot_general_dynamic_rhs_out_dim(%arg0: tensor<4x4x256xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x?xf32> {
%0 = "mhlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2],
    rhs_contracting_dimensions = [2]
  >} : (tensor<4x4x256xf32>, tensor<4x?x256xf32>) -> tensor<4x4x?xf32>
func.return %0 : tensor<4x4x?xf32>

// CHECK-LABEL:     convert_dot_general_dynamic_rhs_out_dim
// CHECK:           %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x4x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT:      %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<4x?x256xf32>) -> tensor<4x256x?xf32>
// CHECK-NEXT:      %2 = mhlo.reshape %0 : (tensor<4x4x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT:      %3 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %6 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %10 = "tfl.concatenation"(%9, %8, %7) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %11 = mhlo.dynamic_reshape %1, %10 : (tensor<4x256x?xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT:      %12 = "tfl.batch_matmul"(%2, %11) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x4x256xf32>, tensor<4x256x?xf32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      %13 = "tfl.shape"(%arg0) : (tensor<4x4x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %14 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %15 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %16 = "tfl.gather"(%13, %15) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<3xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %17 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NEXT:      %18 = "tfl.gather"(%14, %17) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<3xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT:      %19 = "tfl.concatenation"(%16, %18) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %20 = mhlo.dynamic_reshape %12, %19 : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      return %20 : tensor<4x4x?xf32>
}

func.func @convert_dot_general_dynamic_batch_dim(%arg0: tensor<2x?x2x3xf32>, %arg1: tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32> {
%0 = "mhlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0, 1],
    rhs_batching_dimensions = [0, 1],
    lhs_contracting_dimensions = [3],
    rhs_contracting_dimensions = [3]
  >} : (tensor<2x?x2x3xf32>, tensor<2x?x4x3xf32>) -> tensor<2x?x2x4xf32>
func.return %0 : tensor<2x?x2x4xf32>

// CHECK-LABEL:     convert_dot_general_dynamic_batch_dim
// CHECK:           %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x?x2x3xf32>) -> tensor<2x?x2x3xf32>
// CHECK-NEXT:      %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<2x?x4x3xf32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT:      %2 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %3 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %6 = "tfl.unsorted_segment_prod"(%2, %3, %5) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%2, %4, %5) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %9 = "tfl.gather"(%2, %8) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %10 = "tfl.concatenation"(%9, %6, %7) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %11 = mhlo.dynamic_reshape %0, %10 : (tensor<2x?x2x3xf32>, tensor<4xi32>) -> tensor<2x?x2x3xf32>
// CHECK-NEXT:      %12 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %13 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %14 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %15 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %16 = "tfl.unsorted_segment_prod"(%12, %13, %15) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %17 = "tfl.unsorted_segment_prod"(%12, %14, %15) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %18 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %19 = "tfl.gather"(%12, %18) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %20 = "tfl.concatenation"(%19, %17, %16) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %21 = mhlo.dynamic_reshape %1, %20 : (tensor<2x?x3x4xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT:      %22 = "tfl.batch_matmul"(%11, %21) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x?x2x3xf32>, tensor<2x?x3x4xf32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT:      %23 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %24 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %25 = "tfl.pseudo_const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:      %26 = "tfl.gather"(%23, %25) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %27 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NEXT:      %28 = "tfl.gather"(%24, %27) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT:      %29 = "tfl.concatenation"(%26, %28) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %30 = mhlo.dynamic_reshape %22, %29 : (tensor<2x?x2x4xf32>, tensor<4xi32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT:      return %30 : tensor<2x?x2x4xf32>
}


func.func @convert_dot_general_dynamic_lhs_rhs_out_dims(%arg0: tensor<2x2x?x3xf32>, %arg1: tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32> {
%0 = "mhlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [3],
    rhs_contracting_dimensions = [3]
  >} : (tensor<2x2x?x3xf32>, tensor<2x4x?x3xf32>) -> tensor<2x2x?x4x?xf32>
func.return %0 : tensor<2x2x?x4x?xf32>

// CHECK-LABEL:     convert_dot_general_dynamic_lhs_rhs_out_dims
// CHECK:           %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x2x?x3xf32>) -> tensor<2x2x?x3xf32>
// CHECK-NEXT:      %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<2x4x?x3xf32>) -> tensor<2x3x4x?xf32>
// CHECK-NEXT:      %2 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %3 = "tfl.pseudo_const"() {value = dense<[-1, 0, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %6 = "tfl.unsorted_segment_prod"(%2, %3, %5) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%2, %4, %5) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.concatenation"(%8, %6, %7) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %10 = mhlo.dynamic_reshape %0, %9 : (tensor<2x2x?x3xf32>, tensor<3xi32>) -> tensor<2x?x3xf32>
// CHECK-NEXT:      %11 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %12 = "tfl.pseudo_const"() {value = dense<[-1, 0, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %13 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %14 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %15 = "tfl.unsorted_segment_prod"(%11, %12, %14) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %16 = "tfl.unsorted_segment_prod"(%11, %13, %14) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %17 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %18 = "tfl.concatenation"(%17, %16, %15) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %19 = mhlo.dynamic_reshape %1, %18 : (tensor<2x3x4x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK-NEXT:      %20 = "tfl.batch_matmul"(%10, %19) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x?x3xf32>, tensor<2x3x?xf32>) -> tensor<2x?x?xf32>
// CHECK-NEXT:      %21 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %22 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %23 = "tfl.pseudo_const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:      %24 = "tfl.gather"(%21, %23) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %25 = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %26 = "tfl.gather"(%22, %25) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %27 = "tfl.concatenation"(%24, %26) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<3xi32>, tensor<2xi32>) -> tensor<5xi32>
// CHECK-NEXT:      %28 = mhlo.dynamic_reshape %20, %27 : (tensor<2x?x?xf32>, tensor<5xi32>) -> tensor<2x2x?x4x?xf32>
// CHECK-NEXT:      return %28 : tensor<2x2x?x4x?xf32>

}

func.func @convert_dot_general_dynamic_contracting_dim(%arg0: tensor<4x4x?xf32>, %arg1: tensor<4x?x256xf32>) -> tensor<4x4x256xf32> {
%0 = "mhlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #mhlo.dot<
    lhs_batching_dimensions = [0],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2],
    rhs_contracting_dimensions = [1]
  >} : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
func.return %0 : tensor<4x4x256xf32>

// CHECK-LABEL:     convert_dot_general_dynamic_contracting_dim
// CHECK:           %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x4x?xf32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      %1 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x?x256xf32>) -> tensor<4x?x256xf32>
// CHECK-NEXT:      %2 = "tfl.shape"(%arg0) : (tensor<4x4x?xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %3 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %6 = "tfl.unsorted_segment_prod"(%2, %3, %5) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%2, %4, %5) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.concatenation"(%8, %6, %7) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %10 = mhlo.dynamic_reshape %0, %9 : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      %11 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %12 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %13 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %14 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %15 = "tfl.unsorted_segment_prod"(%11, %12, %14) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %16 = "tfl.unsorted_segment_prod"(%11, %13, %14) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %17 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %18 = "tfl.concatenation"(%17, %16, %15) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %19 = mhlo.dynamic_reshape %1, %18 : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x?x256xf32>
// CHECK-NEXT:      %20 = "tfl.batch_matmul"(%10, %19) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT:      %21 = mhlo.reshape %20 : (tensor<4x4x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT:      return %21 : tensor<4x4x256xf32>
}
