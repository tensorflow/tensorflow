// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
// CHECK-LABEL: main
// CHECK: return %arg0 : tensor<5x7xf32>
}

// - transpose
//
func.func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>

// CHECK-LABEL:   transpose_2d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK-NEXT:    return %2 : tensor<3x2xf32>
}

func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>

// CHECK-LABEL:   transpose_3d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:    return %2 : tensor<3x2x1xf32>
}

func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>

// CHECK-LABEL:   transpose_dynamic_2d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<4x?xf32>
// CHECK-NEXT:    return %2 : tensor<4x?xf32>
}

func.func @transpose_unranked_2d(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>

// CHECK-LABEL:   transpose_unranked_2d
// CHECK-NEXT:    %0 = "tfl.pseudo_const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:    %1 = "tfl.cast"(%0) : (tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %2 = "tfl.transpose"(%arg0, %1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
// CHECK-NEXT:    return %2 : tensor<*xf32>
}

// - reshape
//
func.func @reshape(%arg0: tensor<4x6xf32>) -> tensor<2x2x6xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<4x6xf32>) -> tensor<2x2x6xf32>
  func.return %0 : tensor<2x2x6xf32>

// CHECK-LABEL:     reshape
// CHECK-NEXT:      %[[I64SHAPE:.*]] = arith.constant
// CHECK-NEXT:      %[[I32SHAPE:.*]] = "tfl.cast"(%[[I64SHAPE]]) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %[[RESHAPED:.*]] = "tfl.reshape"(%arg0, %[[I32SHAPE]]) : (tensor<4x6xf32>, tensor<3xi32>) -> tensor<2x2x6xf32>
// CHECK-NEXT:      return %[[RESHAPED]] : tensor<2x2x6xf32>
}


func.func @dynamic_reshape(%arg0: tensor<1x1x1x?xf32>, %arg1: tensor<3xi32>, %arg2: tensor<1xi32>) -> tensor<?xf32> {
  %0 = mhlo.dynamic_reshape %arg0, %arg1 : (tensor<1x1x1x?xf32>, tensor<3xi32>) -> tensor<1x1x?xf32>
  %1 = mhlo.dynamic_reshape %0, %arg2 : (tensor<1x1x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>

// CHECK-LABEL:   dynamic_reshape
// CHECK-NEXT:    %0 = "tfl.cast"(%arg1) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:    %1 = "tfl.reshape"(%arg0, %0) : (tensor<1x1x1x?xf32>, tensor<3xi32>) -> tensor<1x1x?xf32>
// CHECK-NEXT:    %2 = "tfl.cast"(%arg2) : (tensor<1xi32>) -> tensor<1xi32>
// CHECK-NEXT:    %3 = "tfl.reshape"(%1, %2) : (tensor<1x1x?xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK:         return %3 : tensor<?xf32>
}

// - dot_general
//

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
// CHECK:         %[[TRANSPOSED_0:.*]] = "tfl.transpose"
// CHECK:         %[[TRANSPOSED_1:.*]] = "tfl.transpose"
// CHECK:         %[[SHAPE_0:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = "tfl.reshape"(%[[TRANSPOSED_0]], %[[SHAPE_0]])
// CHECK:         %[[SHAPE_1:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_1:.*]] = "tfl.reshape"(%[[TRANSPOSED_1]], %[[SHAPE_1]])
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %[[RESHAPED_1]]) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<3x5x12xf32>, tensor<3x12x4xf32>) -> tensor<3x5x4xf32>
// CHECK:         %[[SHAPE_BMM:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_BMM:.*]] = "tfl.reshape"(%[[BMM_0]], %[[SHAPE_BMM]])
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
// CHECK:         %[[SHAPE_0:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = "tfl.reshape"(%arg0, %[[SHAPE_0]])
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : {{.*}} -> tensor<1x1024xf32>
// CHECK:         %[[SHAPE_BMM:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_BMM:.*]] = "tfl.reshape"(%[[BMM_0]], %[[SHAPE_BMM]])
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
// CHECK:         %[[SHAPE_0:.*]] = "tfl.cast"
// CHECK-NEXT:    %[[RESHAPED_0:.*]] = "tfl.reshape"(%arg0, %[[SHAPE_0]])
// CHECK-NEXT:    %[[BMM_0:.*]] = "tfl.batch_matmul"(%[[RESHAPED_0]], %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : {{.*}} -> tensor<1x8xi32>
// CHECK:         %[[SHAPE_BMM:.*]] = "tfl.cast"
// CHECK:         %[[RESHAPED_BMM:.*]] = "tfl.reshape"(%[[BMM_0]], %[[SHAPE_BMM]])
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
// CHECK:           %0 = "tfl.pseudo_const"() {value = dense<[0, 2, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:      %1 = "tfl.cast"(%0) : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %2 = "tfl.transpose"(%arg1, %1) : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT:      %3 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %6 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %10 = "tfl.concatenation"(%9, %8, %7) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %11 = "tfl.cast"(%10) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %12 = "tfl.reshape"(%2, %11) : (tensor<4x256x?xf32>, tensor<3xi32>) -> tensor<4x256x?xf32>
// CHECK-NEXT:      %13 = "tfl.batch_matmul"(%arg0, %12) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x4x256xf32>, tensor<4x256x?xf32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      %14 = "tfl.shape"(%arg0) : (tensor<4x4x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %15 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %16 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %17 = "tfl.gather"(%14, %16) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<3xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %18 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NEXT:      %19 = "tfl.gather"(%15, %18) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<3xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT:      %20 = "tfl.concatenation"(%17, %19) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %21 = "tfl.cast"(%20) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %22 = "tfl.reshape"(%13, %21) : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      return %22 : tensor<4x4x?xf32>
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
// CHECK:           %0 = "tfl.pseudo_const"() {value = dense<[0, 1, 3, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK-NEXT:      %1 = "tfl.cast"(%0) : (tensor<4xi64>) -> tensor<4xi32>
// CHECK-NEXT:      %2 = "tfl.transpose"(%arg1, %1) : (tensor<2x?x4x3xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT:      %3 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %6 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %10 = "tfl.gather"(%3, %9) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %11 = "tfl.concatenation"(%10, %7, %8) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %12 = "tfl.cast"(%11) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %13 = "tfl.reshape"(%arg0, %12) : (tensor<2x?x2x3xf32>, tensor<4xi32>) -> tensor<2x?x2x3xf32>
// CHECK-NEXT:      %14 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %15 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %16 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %17 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %18 = "tfl.unsorted_segment_prod"(%14, %15, %17) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %19 = "tfl.unsorted_segment_prod"(%14, %16, %17) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %20 = "tfl.pseudo_const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %21 = "tfl.gather"(%14, %20) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %22 = "tfl.concatenation"(%21, %19, %18) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %23 = "tfl.cast"(%22) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %24 = "tfl.reshape"(%2, %23) : (tensor<2x?x3x4xf32>, tensor<4xi32>) -> tensor<2x?x3x4xf32>
// CHECK-NEXT:      %25 = "tfl.batch_matmul"(%13, %24) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x?x2x3xf32>, tensor<2x?x3x4xf32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT:      %26 = "tfl.shape"(%arg0) : (tensor<2x?x2x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %27 = "tfl.shape"(%arg1) : (tensor<2x?x4x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %28 = "tfl.pseudo_const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:      %29 = "tfl.gather"(%26, %28) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %30 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NEXT:      %31 = "tfl.gather"(%27, %30) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<1xi64>) -> tensor<1xi32>
// CHECK-NEXT:      %32 = "tfl.concatenation"(%29, %31) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<3xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %33 = "tfl.cast"(%32) : (tensor<4xi32>) -> tensor<4xi32>
// CHECK-NEXT:      %34 = "tfl.reshape"(%25, %33) : (tensor<2x?x2x4xf32>, tensor<4xi32>) -> tensor<2x?x2x4xf32>
// CHECK-NEXT:      return %34 : tensor<2x?x2x4xf32>
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
// CHECK-NEXT:      %0 = "tfl.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK-NEXT:      %1 = "tfl.cast"(%0) : (tensor<4xi64>) -> tensor<4xi32>
// CHECK-NEXT:      %2 = "tfl.transpose"(%arg1, %1) : (tensor<2x4x?x3xf32>, tensor<4xi32>) -> tensor<2x3x4x?xf32>
// CHECK-NEXT:      %3 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %4 = "tfl.pseudo_const"() {value = dense<[-1, 0, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %5 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %6 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %7 = "tfl.unsorted_segment_prod"(%3, %4, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %8 = "tfl.unsorted_segment_prod"(%3, %5, %6) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %9 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %10 = "tfl.concatenation"(%9, %7, %8) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %11 = "tfl.cast"(%10) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %12 = "tfl.reshape"(%arg0, %11) : (tensor<2x2x?x3xf32>, tensor<3xi32>) -> tensor<2x?x3xf32>
// CHECK-NEXT:      %13 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %14 = "tfl.pseudo_const"() {value = dense<[-1, 0, 0, -1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %15 = "tfl.pseudo_const"() {value = dense<[-1, -1, -1, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-NEXT:      %16 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %17 = "tfl.unsorted_segment_prod"(%13, %14, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %18 = "tfl.unsorted_segment_prod"(%13, %15, %16) : (tensor<4xi32>, tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %19 = "tfl.pseudo_const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %20 = "tfl.concatenation"(%19, %18, %17) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %21 = "tfl.cast"(%20) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %22 = "tfl.reshape"(%2, %21) : (tensor<2x3x4x?xf32>, tensor<3xi32>) -> tensor<2x3x?xf32>
// CHECK-NEXT:      %23 = "tfl.batch_matmul"(%12, %22) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<2x?x3xf32>, tensor<2x3x?xf32>) -> tensor<2x?x?xf32>
// CHECK-NEXT:      %24 = "tfl.shape"(%arg0) : (tensor<2x2x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %25 = "tfl.shape"(%arg1) : (tensor<2x4x?x3xf32>) -> tensor<4xi32>
// CHECK-NEXT:      %26 = "tfl.pseudo_const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT:      %27 = "tfl.gather"(%24, %26) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:      %28 = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-NEXT:      %29 = "tfl.gather"(%25, %28) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4xi32>, tensor<2xi64>) -> tensor<2xi32>
// CHECK-NEXT:      %30 = "tfl.concatenation"(%27, %29) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<3xi32>, tensor<2xi32>) -> tensor<5xi32>
// CHECK-NEXT:      %31 = "tfl.cast"(%30) : (tensor<5xi32>) -> tensor<5xi32>
// CHECK-NEXT:      %32 = "tfl.reshape"(%23, %31) : (tensor<2x?x?xf32>, tensor<5xi32>) -> tensor<2x2x?x4x?xf32>
// CHECK-NEXT:      return %32 : tensor<2x2x?x4x?xf32>

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
// CHECK-NEXT:      %0 = "tfl.shape"(%arg0) : (tensor<4x4x?xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %1 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %2 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %3 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %4 = "tfl.unsorted_segment_prod"(%0, %1, %3) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %5 = "tfl.unsorted_segment_prod"(%0, %2, %3) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %6 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %7 = "tfl.concatenation"(%6, %4, %5) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %8 = "tfl.cast"(%7) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %9 = "tfl.reshape"(%arg0, %8) : (tensor<4x4x?xf32>, tensor<3xi32>) -> tensor<4x4x?xf32>
// CHECK-NEXT:      %10 = "tfl.shape"(%arg1) : (tensor<4x?x256xf32>) -> tensor<3xi32>
// CHECK-NEXT:      %11 = "tfl.pseudo_const"() {value = dense<[-1, -1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %12 = "tfl.pseudo_const"() {value = dense<[-1, 0, -1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-NEXT:      %13 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT:      %14 = "tfl.unsorted_segment_prod"(%10, %11, %13) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %15 = "tfl.unsorted_segment_prod"(%10, %12, %13) : (tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:      %16 = "tfl.pseudo_const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-NEXT:      %17 = "tfl.concatenation"(%16, %15, %14) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %18 = "tfl.cast"(%17) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:      %19 = "tfl.reshape"(%arg1, %18) : (tensor<4x?x256xf32>, tensor<3xi32>) -> tensor<4x?x256xf32>
// CHECK-NEXT:      %20 = "tfl.batch_matmul"(%9, %19) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x4x?xf32>, tensor<4x?x256xf32>) -> tensor<4x4x256xf32>
// CHECK-NEXT:      return %20 : tensor<4x4x256xf32>
}