// RUN: stablehlo-quant-opt -optimize-int-graph -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @convolution_add_add
func.func @convolution_add_add(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi32>, %bias: tensor<1x2x2x1xi32>
  ) -> tensor<1x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[conv:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi32>
  %1 = mhlo.add %0, %zp_offset : tensor<1x2x2x1xi32>
  %2 = mhlo.add %1, %bias : tensor<1x2x2x1xi32>
  return %2 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_subtract
func.func @convolution_add_subtract(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi32>, %bias: tensor<1x2x2x1xi32>
  ) -> tensor<1x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.subtract %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[conv:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi32>
  %1 = mhlo.add %0, %zp_offset : tensor<1x2x2x1xi32>
  %2 = mhlo.subtract %1, %bias : tensor<1x2x2x1xi32>
  return %2 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_subtract_subtract
func.func @convolution_subtract_subtract(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi32>, %bias: tensor<1x2x2x1xi32>
  ) -> tensor<1x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.subtract %[[conv:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi32>
  %1 = mhlo.subtract %0, %zp_offset : tensor<1x2x2x1xi32>
  %2 = mhlo.subtract %1, %bias : tensor<1x2x2x1xi32>
  return %2 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_subtract_add
func.func @convolution_subtract_add(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi32>, %bias: tensor<1x2x2x1xi32>
  ) -> tensor<1x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.subtract %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.subtract %[[conv:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi32>
  %1 = mhlo.subtract %0, %zp_offset : tensor<1x2x2x1xi32>
  %2 = mhlo.add %1, %bias : tensor<1x2x2x1xi32>
  return %2 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_add_add
func.func @convolution_add_add_add(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi32>, %bias: tensor<1x2x2x1xi32>
  ) -> tensor<1x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined1:.*]] = mhlo.add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[combined2:.*]] = mhlo.add %[[combined1:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[conv:.*]], %[[combined2:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi32>
  %1 = mhlo.add %0, %zp_offset : tensor<1x2x2x1xi32>
  %2 = mhlo.add %1, %bias : tensor<1x2x2x1xi32>
  %3 = mhlo.add %2, %bias : tensor<1x2x2x1xi32>
  return %3 : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_add_i8
func.func @convolution_add_add_i8(
    %lhs: tensor<1x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<1x2x2x1xi8>, %bias: tensor<1x2x2x1xi8>
  ) -> tensor<1x2x2x1xi8> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[conv:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<1x2x2x1xi8>
  %1 = mhlo.add %0, %zp_offset : tensor<1x2x2x1xi8>
  %2 = mhlo.add %1, %bias : tensor<1x2x2x1xi8>
  return %2 : tensor<1x2x2x1xi8>
}

// -----

// CHECK-LABEL: func @convolution_add_add_f32
func.func @convolution_add_add_f32(
    %lhs: tensor<1x3x2x1xf32>, %rhs: tensor<2x1x1x1xf32>,
    %zp_offset: tensor<1x2x2x1xf32>, %bias: tensor<1x2x2x1xf32>
  ) -> tensor<1x2x2x1xf32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = mhlo.add %[[conv:.*]], %[[zp_offset:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[combined:.*]], %[[bias:.*]]
  // CHECK: return %[[result:.*]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<1x3x2x1xf32>, tensor<2x1x1x1xf32>) -> tensor<1x2x2x1xf32>
  %1 = mhlo.add %0, %zp_offset : tensor<1x2x2x1xf32>
  %2 = mhlo.add %1, %bias : tensor<1x2x2x1xf32>
  return %2 : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @dot_general_add_add
func.func @dot_general_add_add(
    %lhs: tensor<2x5x6xi8>, %rhs: tensor<6x8x2xi8>,
    %zp_offset: tensor<2x5x8xi32>, %bias: tensor<2x5x8xi32>
  ) -> tensor<2x5x8xi32> {
  // CHECK-DAG: %[[dot:.*]] = "mhlo.dot_general"
  // CHECK-DAG: %[[combined:.*]] = mhlo.add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = mhlo.add %[[dot:.*]], %[[combined:.*]]
  // CHECK: return %[[result:.*]]
  %0 = "mhlo.dot_general" (%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6xi8>, tensor<6x8x2xi8>
    ) -> tensor<2x5x8xi32>
  %1 = mhlo.add %0, %zp_offset : tensor<2x5x8xi32>
  %2 = mhlo.add %1, %bias : tensor<2x5x8xi32>
  return %2 : tensor<2x5x8xi32>
}
