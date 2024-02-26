// RUN: stablehlo-quant-opt -optimize-int-graph -split-input-file %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @convolution_add_add
func.func @convolution_add_add(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<?x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x1xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  return %2 : tensor<?x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_add_static
func.func @convolution_add_add_static(
    %lhs: tensor<2x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<2x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<2x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<2x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<2x2x2x1xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<2x2x2x1xi32>, tensor<2x2x2x1xi32>) -> tensor<2x2x2x1xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<2x2x2x1xi32>, tensor<1xi32>) ->tensor<2x2x2x1xi32>
  return %2 : tensor<2x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_subtract
func.func @convolution_add_subtract(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<?x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_subtract %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x1xi32>
  %2 = chlo.broadcast_subtract %1, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  return %2 : tensor<?x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_subtract_subtract
func.func @convolution_subtract_subtract(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<?x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_subtract %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi32>
  %1 = chlo.broadcast_subtract %0, %zp_offset : (
      tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x1xi32>
  %2 = chlo.broadcast_subtract %1, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  return %2 : tensor<?x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_subtract_add
func.func @convolution_subtract_add(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<?x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_subtract %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_subtract %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi32>
  %1 = chlo.broadcast_subtract %0, %zp_offset : (
      tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x1xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  return %2 : tensor<?x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_add_add
func.func @convolution_add_add_add(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi32>, %bias: tensor<1xi32>
  ) -> tensor<?x2x2x1xi32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined1:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[combined2:.*]] = chlo.broadcast_add %[[combined1]], %[[bias]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[conv]], %[[combined2]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x1xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  %3 = chlo.broadcast_add %2, %bias : (
      tensor<?x2x2x1xi32>, tensor<1xi32>) ->tensor<?x2x2x1xi32>
  return %3 : tensor<?x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @convolution_add_add_i8
func.func @convolution_add_add_i8(
    %lhs: tensor<?x3x2x1xi8>, %rhs: tensor<2x1x1x1xi8>,
    %zp_offset: tensor<?x2x2x1xi8>, %bias: tensor<1xi8>
  ) -> tensor<?x2x2x1xi8> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[conv]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xi8>, tensor<2x1x1x1xi8>) -> tensor<?x2x2x1xi8>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x2x2x1xi8>, tensor<?x2x2x1xi8>) -> tensor<?x2x2x1xi8>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x2x2x1xi8>, tensor<1xi8>) ->tensor<?x2x2x1xi8>
  return %2 : tensor<?x2x2x1xi8>
}

// -----

// CHECK-LABEL: func @convolution_add_add_f32
func.func @convolution_add_add_f32(
    %lhs: tensor<?x3x2x1xf32>, %rhs: tensor<2x1x1x1xf32>,
    %zp_offset: tensor<?x2x2x1xf32>, %bias: tensor<1xf32>
  ) -> tensor<?x2x2x1xf32> {
  // CHECK-DAG: %[[conv:.*]] = mhlo.convolution
  // CHECK-DAG: %[[combined:.*]] = chlo.broadcast_add %[[conv:.*]], %[[zp_offset:.*]]
  // CHECK-DAG: %[[result:.*]] = chlo.broadcast_add %[[combined:.*]], %[[bias:.*]]
  // CHECK: return %[[result]]
  %0 = mhlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    : (tensor<?x3x2x1xf32>, tensor<2x1x1x1xf32>) -> tensor<?x2x2x1xf32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x2x2x1xf32>, tensor<?x2x2x1xf32>) -> tensor<?x2x2x1xf32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x2x2x1xf32>, tensor<1xf32>) ->tensor<?x2x2x1xf32>
  return %2 : tensor<?x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @dot_general_add_add
func.func @dot_general_add_add(
    %lhs: tensor<?x?x6xi8>, %rhs: tensor<6x8xi8>,
    %zp_offset: tensor<8xi32>, %bias: tensor<8xi32>
  ) -> tensor<?x?x8xi32> {
  // CHECK: %[[dot:.*]] = "mhlo.dot_general"
  // CHECK: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK: %[[result:.*]] = chlo.broadcast_add %[[dot]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = "mhlo.dot_general" (%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<?x?x6xi8>, tensor<6x8xi8>
    ) -> tensor<?x?x8xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<?x?x8xi32>, tensor<8xi32>) -> tensor<?x?x8xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<?x?x8xi32>, tensor<8xi32>) -> tensor<?x?x8xi32>
  return %2 : tensor<?x?x8xi32>
}

// -----

// CHECK-LABEL: func @dot_general_add_add_static
func.func @dot_general_add_add_static(
    %lhs: tensor<2x5x6xi8>, %rhs: tensor<6x8x2xi8>,
    %zp_offset: tensor<2x5x8xi32>, %bias: tensor<2x5x8xi32>
  ) -> tensor<2x5x8xi32> {
  // CHECK: %[[dot:.*]] = "mhlo.dot_general"
  // CHECK: %[[combined:.*]] = chlo.broadcast_add %[[zp_offset:.*]], %[[bias:.*]]
  // CHECK: %[[result:.*]] = chlo.broadcast_add %[[dot]], %[[combined]]
  // CHECK: return %[[result]]
  %0 = "mhlo.dot_general" (%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (
      tensor<2x5x6xi8>, tensor<6x8x2xi8>
    ) -> tensor<2x5x8xi32>
  %1 = chlo.broadcast_add %0, %zp_offset : (
      tensor<2x5x8xi32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  %2 = chlo.broadcast_add %1, %bias : (
      tensor<2x5x8xi32>, tensor<2x5x8xi32>) -> tensor<2x5x8xi32>
  return %2 : tensor<2x5x8xi32>
}
