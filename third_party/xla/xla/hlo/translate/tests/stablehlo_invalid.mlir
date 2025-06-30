// RUN: not hlo-translate -mlir-to-hlo -split-input-file %s 2>&1 | FileCheck %s

// StableHLO ops that has no HLO support. These all must be refined away before
// lowering. See https://openxla.org/stablehlo/dynamism

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: Shape Error: Invalid element type
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [0] {known_expanding_dimensions = array<i64>, known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @main(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<2x2xi32>) -> tensor<1x?x?x16xf32> {
  // CHECK: Shape Error: Invalid element type
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    window_reversal = array<i1: false, false>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}

// -----

func.func @main(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // CHECK: op can't be translated to XLA HLO
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @main(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: op can't be translated to XLA HLO
  %0 = "stablehlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @op_dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: op can't be translated to XLA HLO
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @op_dynamic_reshape(%arg0: tensor<16xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: op can't be translated to XLA HLO
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<16xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @op_unary_einsum(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
  // CHECK: op can't be translated to XLA HLO
  %0 = "stablehlo.unary_einsum"(%arg0) {
    einsum_config = "ab->a"
  } : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
