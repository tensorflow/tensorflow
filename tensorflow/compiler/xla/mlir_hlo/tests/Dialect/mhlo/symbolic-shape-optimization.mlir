// RUN: mlir-hlo-opt %s --split-input-file --symbolic-shape-optimization | \
// RUN: FileCheck %s

// CHECK-LABEL: func @reshape_expand_front
func.func @reshape_expand_front(%arg0: tensor<?x?xf32>) -> tensor<1x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %shape = tensor.from_elements %c1, %d0, %d1 : tensor<3xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<1x?x?xf32>
// CHECK: tensor.expand_shape %arg0 [
// CHECK-SAME: [0, 1], [2]] : tensor<?x?xf32> into tensor<1x?x?xf32>
  func.return %reshape : tensor<1x?x?xf32>
}

// CHECK-LABEL: func @reshape_expand_front_static
func.func @reshape_expand_front_static(%arg0: tensor<2x?xf32>) -> tensor<1x2x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<2x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<2x?xf32>
  %shape = tensor.from_elements %c1, %d0, %d1 : tensor<3xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<2x?xf32>, tensor<3xindex>) -> tensor<1x2x?xf32>
// CHECK: tensor.expand_shape %arg0 [
// CHECK-SAME: [0, 1], [2]] : tensor<2x?xf32> into tensor<1x2x?xf32>
  func.return %reshape : tensor<1x2x?xf32>
}

// -----

// CHECK-LABEL: func @reshape_expand_back
func.func @reshape_expand_back(%arg0: tensor<?x?xf32>) -> tensor<?x?x1x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %shape = tensor.from_elements %d0, %d1, %c1, %c1 : tensor<4xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x1x1xf32>
// CHECK: tensor.expand_shape %arg0 [
// CHECK-SAME: [0], [1, 2, 3]] : tensor<?x?xf32> into tensor<?x?x1x1xf32>
  func.return %reshape : tensor<?x?x1x1xf32>
}

// -----

// CHECK-LABEL: @reshape_expand_scalar
// CHECK-SAME:  %[[ARG:.*]]: tensor<f32>
func.func @reshape_expand_scalar(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[EXPAND:.*]] = tensor.expand_shape %[[ARG]] [] : tensor<f32> into tensor<1x1xf32>
  // CHECK-DAG: %[[RES:.*]] = tensor.cast %[[EXPAND]] : tensor<1x1xf32> to tensor<?x?xf32>
  // CHECK:     return %[[RES]]
  %shape = mhlo.constant dense<1> : tensor<2xi32>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %reshape : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @reshape_collapse_scalar
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>
func.func @reshape_collapse_scalar(%arg0 : tensor<?x?xf32>) -> tensor<f32> {
  %shape = mhlo.constant dense<1> : tensor<0xi32>
  // CHECK-DAG: %[[CASTED_ARG:.*]] = tensor.cast %[[ARG]] : tensor<?x?xf32> to tensor<1x1xf32>
  // CHECK-DAG: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CASTED_ARG]] [] : tensor<1x1xf32> into tensor<f32>
  // CHECK:     return %[[COLLAPSED]]
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?x?xf32>, tensor<0xi32>) -> tensor<f32>
  func.return %reshape : tensor<f32>
}

// -----

// CHECK-LABEL: func @reshape_undefined
func.func @reshape_undefined(%arg0: tensor<?xf32>) -> tensor<1x1x1xf32> {
  // CHECK: mhlo.dynamic_reshape
  %c1 = arith.constant 1 : index
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<?xf32>, tensor<3xindex>) -> tensor<1x1x1xf32>
  func.return %reshape : tensor<1x1x1xf32>
}

// -----

// CHECK-LABEL: @shape_expansion
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x1xi64>
func.func @shape_expansion(%arg : tensor<?x1xi64>) -> tensor<?x1x1xi64> {
  // CHECK-DAG: %[[RES:.*]] = tensor.expand_shape %[[ARG]] {{\[}}[0], [1, 2]{{\]}} : tensor<?x1xi64> into tensor<?x1x1xi64>
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg, %c0 : tensor<?x1xi64>
  %shape = tensor.from_elements %d0, %c1, %c1 : tensor<3xindex>
  %result = "mhlo.dynamic_reshape"(%arg, %shape)
      : (tensor<?x1xi64>, tensor<3xindex>) -> tensor<?x1x1xi64>
  func.return %result : tensor<?x1x1xi64>
}

// -----

// CHECK-LABEL: @shape_collapse_and_expansion
// CHECK-SAME:  %[[ARG:.*]]: tensor<3x?x1xi64>
func.func @shape_collapse_and_expansion(%arg : tensor<3x?x1xi64>)
    -> tensor<?x1x1xi64> {
  // CHECK: %[[EED:.*]] = tensor.expand_shape %[[ARG]] {{\[}}[0], [1], [2, 3]{{\]}} : tensor<3x?x1xi64> into tensor<3x?x1x1xi64>
  // CHECK: %[[CED:.*]] = tensor.collapse_shape %[[EED]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<3x?x1x1xi64> into tensor<?x1x1xi64>
  // CHECK: return %[[CED]]
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %d1 = tensor.dim %arg, %c1 : tensor<3x?x1xi64>
  %three_d1 = arith.muli %c3, %d1 : index
  %15 = tensor.from_elements %three_d1, %c1, %c1 : tensor<3xindex>
  %16 = "mhlo.dynamic_reshape"(%arg, %15)
      : (tensor<3x?x1xi64>, tensor<3xindex>) -> tensor<?x1x1xi64>
  func.return %16 : tensor<?x1x1xi64>
}

// -----

// CHECK-LABEL: @shape_collapse_and_expansion_w_cast
// CHECK-SAME:  %[[ARG:.*]]: tensor<16x8x?x?xf32>
func.func @shape_collapse_and_expansion_w_cast(%arg0: tensor<16x8x?x?xf32>) -> tensor<16x4x?x?xf32> {
  // CHECK-DAG: %[[EED:.*]] = tensor.expand_shape %[[ARG]] {{\[}}[0], [1, 2], [3], [4]{{\]}} : tensor<16x8x?x?xf32> into tensor<16x4x2x?x?xf32>
  // CHECK-DAG: %[[CED:.*]] = tensor.collapse_shape %[[EED]] {{\[}}[0], [1], [2], [3, 4]{{\]}} : tensor<16x4x2x?x?xf32> into tensor<16x4x2x?xf32>
  // CHECK-DAG: %[[RES:.*]] = tensor.cast %[[CED]]
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %1 = tensor.dim %arg0, %c2 : tensor<16x8x?x?xf32>
  %2 = tensor.dim %arg0, %c3 : tensor<16x8x?x?xf32>
  %4 = arith.muli %1, %2 : index
  %5 = tensor.from_elements %c16, %c4, %c2, %4 : tensor<4xindex>
  %6 = "mhlo.dynamic_reshape"(%arg0, %5)  : (tensor<16x8x?x?xf32>, tensor<4xindex>) -> tensor<16x4x?x?xf32>
  func.return %6 : tensor<16x4x?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_reshape_to_collapse_shape
// CHECK-SAME: %[[ARG:.*]]: tensor<1x4x?x64x?x8x1x1xf32>
func.func @dynamic_reshape_to_collapse_shape(%arg0 : tensor<1x4x?x64x?x8x1x1xf32>)
    -> tensor<?x?x8xf32> {
  // CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[ARG]] {{\[}}[0, 1, 2], [3, 4], [5, 6, 7]{{\]}}
  // CHECK: return %[[RESULT]]
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c64_i32 = arith.constant 64 : i32
  %d2 = tensor.dim %arg0, %c2 : tensor<1x4x?x64x?x8x1x1xf32>
  %d4 = tensor.dim %arg0, %c4 : tensor<1x4x?x64x?x8x1x1xf32>
  %d2_i32 = arith.index_cast %d2 : index to i32
  %d4_i32 = arith.index_cast %d4 : index to i32
  %s0 = arith.muli %c4_i32, %d2_i32 : i32
  %s1 = arith.muli %c64_i32, %d4_i32 : i32
  %shape = tensor.from_elements %s0, %s1, %c8_i32 : tensor<3xi32>
  %result = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<1x4x?x64x?x8x1x1xf32>, tensor<3xi32>) -> tensor<?x?x8xf32>
  func.return %result : tensor<?x?x8xf32>
}

// -----

// CHECK-LABEL: @expansion_unit_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<1x?x1xi64>
func.func @expansion_unit_dims(%arg0: tensor<1x?x1xi64>) -> tensor<1x1x?x1xi64> {
  // CHECK-DAG: %[[RES:.*]] = tensor.expand_shape %[[ARG]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<1x?x1xi64> into tensor<1x1x?x1xi64>
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<1x?x1xi64>
  %1 = tensor.from_elements %c1, %c1, %0, %c1 : tensor<4xindex>
  %2 = "mhlo.dynamic_reshape"(%arg0, %1)
      : (tensor<1x?x1xi64>, tensor<4xindex>) -> tensor<1x1x?x1xi64>
  func.return %2 : tensor<1x1x?x1xi64>
}

// -----

// CHECK-LABEL: @multiple_reductions_and_reshape
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?x?xi64>
func.func @multiple_reductions_and_reshape(%arg0: tensor<?x?x?x?xi64>) -> tensor<1x1x1x1xi64> {
  // CHECK: %[[RED0:.*]] = mhlo.reduce(%[[ARG]]
  // CHECK: %[[RED0_:.*]] = tensor.expand_shape %[[RED0]] {{\[}}[0], [1], [2, 3]{{\]}} : tensor<?x?x?xi64> into tensor<?x?x?x1xi64>
  // CHECK: %[[RED1:.*]] = mhlo.reduce(%[[RED0_]]
  // CHECK: %[[RED1_:.*]] = tensor.expand_shape %[[RED1]] {{\[}}[0, 1, 2], [3]{{\]}} : tensor<?x1xi64> into tensor<1x1x?x1xi64>
  // CHECK: %[[RED2:.*]] = mhlo.reduce(%[[RED1_]]
  // TODO(b/225204462): This should also become a shape expansion.
  // CHECK: %[[RED2_:.*]] = mhlo.reshape %[[RED2]] : (tensor<1xi64>) -> tensor<1x1x1x1xi64>
  // CHECK: return %[[RED2_]]
  %0 = mhlo.constant dense<9223372036854775807> : tensor<i64>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %1 = mhlo.constant dense<1> : tensor<i64>
  %2 = mhlo.reduce(%arg0 init: %0)
      applies mhlo.minimum across dimensions = [3]
      : (tensor<?x?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi64>
  %3 = tensor.dim %2, %c0 : tensor<?x?x?xi64>
  %4 = tensor.dim %2, %c1 : tensor<?x?x?xi64>
  %5 = tensor.dim %2, %c2 : tensor<?x?x?xi64>
  %6 = tensor.from_elements %3, %4, %5, %c1 : tensor<4xindex>
  %7 = "mhlo.dynamic_reshape"(%2, %6)
      : (tensor<?x?x?xi64>, tensor<4xindex>) -> tensor<?x?x?x1xi64>
  %8 = mhlo.reduce(%7 init: %0)
      applies mhlo.minimum across dimensions = [0, 1]
      : (tensor<?x?x?x1xi64>, tensor<i64>) -> tensor<?x1xi64>
  %9 = tensor.dim %8, %c0 : tensor<?x1xi64>
  %10 = tensor.from_elements %c1, %9, %c1 : tensor<3xindex>
  %11 = "mhlo.dynamic_reshape"(%8, %10)
      : (tensor<?x1xi64>, tensor<3xindex>) -> tensor<1x?x1xi64>
  %12 = tensor.dim %11, %c1 : tensor<1x?x1xi64>
  %13 = tensor.from_elements %c1, %c1, %12, %c1 : tensor<4xindex>
  %14 = "mhlo.dynamic_reshape"(%8, %13)
      : (tensor<?x1xi64>, tensor<4xindex>) -> tensor<1x1x?x1xi64>
  %15 = mhlo.reduce(%14 init: %1)
      applies mhlo.multiply across dimensions = [0, 1, 2]
      : (tensor<1x1x?x1xi64>, tensor<i64>) -> tensor<1xi64>
  %16 = "mhlo.reshape"(%15) : (tensor<1xi64>) -> tensor<1x1x1x1xi64>
  func.return %16 : tensor<1x1x1x1xi64>
}

// -----

// CHECK-LABEL: func @compute_reshape_shape
func.func @compute_reshape_shape(%arg0: tensor<?x?xf32>, %arg1: index)
    -> tensor<2xi32> {
  %shape = shape.shape_of %arg0: tensor<?x?xf32> -> tensor<2xindex>
  %casted = arith.index_cast %shape : tensor<2xindex> to tensor<2xi32>
  %mul = mhlo.multiply %casted, %casted : tensor<2xi32>
// CHECK:  %[[MUL:.*]] = mhlo.multiply
  %crs = mhlo.compute_reshape_shape %arg1, %mul
      : (index, tensor<2xi32>) -> tensor<2xi32>
  func.return %crs : tensor<2xi32>
// CHECK: return %[[MUL]] : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @compute_reshape_shape
func.func @compute_reshape_shape(%arg0: tensor<2xi32>, %arg1: index)
    -> tensor<2xi32> {
  %mul = mhlo.multiply %arg0, %arg0 : tensor<2xi32>
  %crs = mhlo.compute_reshape_shape %arg1, %mul
      : (index, tensor<2xi32>) -> tensor<2xi32>
// CHECK: mhlo.compute_reshape_shape
  func.return %crs : tensor<2xi32>
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func.func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c512 = arith.constant 512 : index
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x8x?x64xf32>
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %s1_ = tensor.from_elements %dim02, %dim00, %c512 : tensor<3xindex>
  %s1 = arith.index_cast %s1_ : tensor<3xindex> to tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : (index, tensor<3xi32>) -> !shape.witness
  func.return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable_less_obvious
func.func @redundant_cstr_reshapable_less_obvious(%arg0 : tensor<?x4x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x4x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x4x?x64xf32>
  %dim00_twice = arith.muli %c2, %dim00 : index
  %dim00_twice_ = arith.index_cast %dim00_twice : index to i32
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x4x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %dim00_twice_, %c128 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : (index, tensor<3xi32>) -> !shape.witness
  func.return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func.func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %cminus1, %c128 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : (index, tensor<3xi32>) -> !shape.witness
  func.return %w : !shape.witness
}

// -----

// CHECK-LABEL: @nonredundant_cstr_reshapable
func.func @nonredundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = mhlo.cstr_reshapable %{{.*}}, %{{.*}}
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c42 = arith.constant 42 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %cminus1, %c42 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : (index, tensor<3xi32>) -> !shape.witness
  func.return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func.func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x8x?x64xf32>
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim00_ = arith.index_cast %dim00 : index to i32
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %dim00_ , %c64, %cminus1 : tensor<4xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : (index, tensor<4xi32>) -> !shape.witness
  func.return %w : !shape.witness
}

// -----

// CHECK-LABEL: func @reshape_integration(
// CHECK-SAME:      %arg0: tensor<512x512xf32>,
// CHECK-SAME:      %arg1: tensor<?x8x?x64xf32>,
// CHECK-SAME:      %[[DYN_SHAPE:.*]]: tensor<4xi32>,
// CHECK-SAME:      %arg3: tensor<512xf32>,
// CHECK-SAME:      %arg4: tensor<?x?x512xf32>,
// CHECK-SAME:      %arg5: tensor<512xf32>,
// CHECK-SAME:      %arg6: tensor<512xf32>,
// CHECK-SAME:      %arg7: tensor<512x2048xf32>,
// CHECK-SAME:      %arg8: tensor<2048xf32>,
// CHECK-SAME:      %arg9: tensor<2048x512xf32>,
// CHECK-SAME:      %arg10: tensor<512xf32>,
// CHECK-SAME:      %arg11: tensor<512xf32>,
// CHECK-SAME:      %arg12: tensor<512xf32>)
func.func @reshape_integration(%arg0: tensor<512x512xf32>,
    %arg1: tensor<?x8x?x64xf32>, %arg2: tensor<4xi32>, %arg3: tensor<512xf32>,
    %arg4: tensor<?x?x512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>,
    %arg7: tensor<512x2048xf32>, %arg8: tensor<2048xf32>,
    %arg9: tensor<2048x512xf32>, %arg10: tensor<512xf32>,
    %arg11: tensor<512xf32>, %arg12: tensor<512xf32>) -> tensor<?x512xf32> {
  %0 = mhlo.constant dense<512> : tensor<1xi32>
  %1 = shape.shape_of %arg1 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %2 = shape.num_elements %1 : tensor<4xindex> -> index
  // CHECK: %[[W:.*]] = mhlo.cstr_reshapable
  %3 = mhlo.cstr_reshapable %2, %arg2 : (index, tensor<4xi32>) -> !shape.witness
  // CHECK: shape.assuming %[[W]]
  %4 = shape.assuming %3 -> (tensor<?x8x?x64xf32>) {
    // CHECK: %[[SHAPE:.*]] = mhlo.compute_reshape_shape %{{.*}}, %[[DYN_SHAPE]]
    %20 = mhlo.compute_reshape_shape %2, %arg2
        : (index, tensor<4xi32>) -> tensor<4xi32>
    // CHECK: mhlo.dynamic_reshape %arg1, %[[SHAPE]]
    %21 = "mhlo.dynamic_reshape"(%arg1, %20)
        : (tensor<?x8x?x64xf32>, tensor<4xi32>) -> tensor<?x8x?x64xf32>
    // CHECK: shape.assuming_yield
    shape.assuming_yield %21 : tensor<?x8x?x64xf32>
  }
  %5 = "mhlo.transpose"(%4) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>}
      : (tensor<?x8x?x64xf32>) -> tensor<?x?x8x64xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 1, 3, 2]>
      : tensor<4xi64>} : (tensor<?x?x8x64xf32>) -> tensor<?x?x64x8xf32>
  %7 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %8 = arith.index_cast %7 : tensor<4xindex> to tensor<4xi32>
  %9 = "mhlo.slice"(%8) {limit_indices = dense<1> : tensor<1xi64>,
      start_indices = dense<0> : tensor<1xi64>,
      strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %10 = "mhlo.reshape"(%9) : (tensor<1xi32>) -> tensor<i32>
  %11 = "mhlo.slice"(%8) {limit_indices = dense<2> : tensor<1xi64>,
      start_indices = dense<1> : tensor<1xi64>,
      strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %12 = "mhlo.reshape"(%11) : (tensor<1xi32>) -> tensor<i32>
  %13 = mhlo.multiply %10, %12 : tensor<i32>
  %14 = "mhlo.reshape"(%13) : (tensor<i32>) -> tensor<1xi32>
  %15 = "mhlo.concatenate"(%14, %0) {dimension = 0 : i64}
      : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %16 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %17 = shape.num_elements %16 : tensor<4xindex> -> index
  // CHECK-NOT: cstr_reshapable
  %18 = mhlo.cstr_reshapable %17, %15 : (index, tensor<2xi32>) -> !shape.witness
  // CHECK-NOT: assuming
  %19 = shape.assuming %18 -> (tensor<?x512xf32>) {
    // CHECK-NOT: compute_reshape_shape
    %20 = mhlo.compute_reshape_shape %17, %15
        : (index, tensor<2xi32>) -> tensor<2xi32>
    // CHECK: tensor.collapse_shape
    %21 = "mhlo.dynamic_reshape"(%6, %20)
        : (tensor<?x?x64x8xf32>, tensor<2xi32>) -> tensor<?x512xf32>
    // CHECK-NOT: assuming_yield
    shape.assuming_yield %21 : tensor<?x512xf32>
  }
  func.return %19 : tensor<?x512xf32>
}

// -----

// CHECK-LABEL: @optimize_1dx1d_constraint
func.func @optimize_1dx1d_constraint(
  %arg0: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> !shape.witness {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  func.return %2: !shape.witness
}

// -----

// CHECK-LABEL: @optimize_1dx1d_constraint_with_static_shape
func.func @optimize_1dx1d_constraint_with_static_shape(
  %arg0: tensor<?xf32>
    {rt.symbolic_shape = dense<[10]> : tensor<1xi64>},
  %arg1: tensor<10xf32>
) -> !shape.witness {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<10xf32> -> tensor<1xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  func.return %2: !shape.witness
}

// -----

// CHECK-LABEL: @optimize_1dx1d_constraint_with_const_shape
func.func @optimize_1dx1d_constraint_with_const_shape(
  %arg0: tensor<512xf32>,
  %arg1: tensor<?x512xf32>
    {rt.symbolic_shape = dense<[-2,512]> : tensor<2xi64>}
) -> !shape.witness {
  %0 = shape.const_shape [512] : tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?x512xf32> -> tensor<2xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<2xindex>
  func.return %2: !shape.witness
}

// -----

// CHECK-LABEL: @optimize_1dx1d_bcast
func.func @optimize_1dx1d_bcast(
    %arg0: tensor<?xf32> {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
    %arg1: tensor<?xf32> {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>})
    -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
      -> tensor<1xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<0>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
      {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
      : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %3: tensor<?xf32>
}

// -----

// CHECK-LABEL: @optimize_1dx2d_bcast_const_shape
func.func @optimize_1dx2d_bcast_const_shape(
    %arg0: tensor<512xf32>,
    %arg1: tensor<?x512xf32>
    {rt.symbolic_shape = dense<[-2, 512]> : tensor<2xi64>})
    -> tensor<?x512xf32> {
  %0 = shape.const_shape [512] : tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?x512xf32> -> tensor<2xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<2xindex>
      -> tensor<2xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<0>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
      {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
      : (tensor<512xf32>, tensor<2xindex>) -> tensor<?x512xf32>
  func.return %3: tensor<?x512xf32>
}

// -----

// CHECK-LABEL: @optimize_1dx1dx1d_bcast
func.func @optimize_1dx1dx1d_bcast(
    %arg0: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
    %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
    %arg2: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>}) -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.shape_of %arg2 : tensor<?xf32> -> tensor<1xindex>
  %3 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
      -> tensor<1xindex>
  %4 = shape.broadcast %3, %2 : tensor<1xindex>, tensor<1xindex>
      -> tensor<1xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<0>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %4)
      {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
      : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %5: tensor<?xf32>
}

// -----

// CHECK-LABEL: @optimize_2dx1d_bcast
func.func @optimize_2dx1d_bcast(
    %arg0: tensor<10x?xf32>
    {rt.symbolic_shape = dense<[10, -2]> : tensor<2xi64>},
    %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>})
    -> (tensor<10x?xf32>, tensor<10x?xf32>) {
  %0 = shape.shape_of %arg0 : tensor<10x?xf32> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<1xindex>
      -> tensor<2xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<[0, 1]>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
      {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
      : (tensor<10x?xf32>, tensor<2xindex>) -> tensor<10x?xf32>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<0>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2)
      {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<10x?xf32>
  func.return %3, %4: tensor<10x?xf32>, tensor<10x?xf32>
}

// -----

// CHECK-LABEL: @optimize_3dx3d_bcast
func.func @optimize_3dx3d_bcast(
    %arg0: tensor<?x1x?xf32>
    {rt.symbolic_shape = dense<[-2, 1, -3]> : tensor<3xi64>},
    %arg1: tensor<1x?x1xf32>
    {rt.symbolic_shape = dense<[1, -4, 1]> : tensor<3xi64>})
    -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %0 = shape.shape_of %arg0 : tensor<?x1x?xf32> -> tensor<3xindex>
  %1 = shape.shape_of %arg1 : tensor<1x?x1xf32> -> tensor<3xindex>
  %2 = shape.broadcast %0, %1 : tensor<3xindex>, tensor<3xindex>
      -> tensor<3xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<[0, 2]>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<?x1x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<1>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2)
      {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
      : (tensor<1x?x1xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %3, %4: tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @optimize_10d_all_cases
func.func @optimize_10d_all_cases(
    %arg0: tensor<1x1x1x8x8x8x?x?x?x?xf32>
    {rt.symbolic_shape = dense<[1, 1,  1, 8, 8,  8, -2, -3, -4, -5]>
    : tensor<10xi64>},
    %arg1: tensor<1x8x?x1x8x?x1x8x?x?xf32>
    {rt.symbolic_shape = dense<[1, 8, -6, 1, 8, -7,  1,  8, -8, -5]>
    : tensor<10xi64>}) -> tensor<?x?x?x?x?x?x?x?x?x?xf32> {
  %0 = shape.shape_of %arg0 : tensor<1x1x1x8x8x8x?x?x?x?xf32>
      -> tensor<10xindex>
  %1 = shape.shape_of %arg1 : tensor<1x8x?x1x8x?x1x8x?x?xf32>
      -> tensor<10xindex>
  %2 = shape.broadcast %0, %1 : tensor<10xindex>, tensor<10xindex>
      -> tensor<10xindex>
  // CHECK:      mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: known_expanding_dimensions = dense<1>
  // CHECK-SAME: known_nonexpanding_dimensions = dense<[0, 3, 4, 5, 6, 9]>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
      {broadcast_dimensions = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]>
      : tensor<10xi64>}
      : (tensor<1x1x1x8x8x8x?x?x?x?xf32>, tensor<10xindex>)
      -> tensor<?x?x?x?x?x?x?x?x?x?xf32>
  func.return %3: tensor<?x?x?x?x?x?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @empty_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>
func.func @empty_bcast(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<0xindex> {
  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<> : tensor<0xindex>
  // CHECK:     return %[[SHAPE]]
  %0 = shape.shape_of %arg0 : tensor<f32> -> tensor<0xindex>
  %1 = shape.shape_of %arg1 : tensor<f32> -> tensor<0xindex>
  %2 = shape.broadcast %0, %1 : tensor<0xindex>, tensor<0xindex>
      -> tensor<0xindex>
  func.return %2 : tensor<0xindex>
}

// -----

// CHECK-LABEL: @simplifiable_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x1x1x4x?x?x1xf32>
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x8x1x?x1x?xf32>
func.func @simplifiable_bcast(
    %arg0 : tensor<?x1x1x4x?x?x1xf32>
    {rt.symbolic_shape = dense<[-2, 1, 1, 4, -2, -3,  1]> : tensor<7xi64>},
    %arg1 : tensor<1x8x1x?x1x?xf32>
    {rt.symbolic_shape = dense<[    1, 8, 1, -2,  1, -4]> : tensor<6xi64>})
    -> tensor<7xindex> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S0D0:.*]] = tensor.extract %[[S0]][%[[C0]]]
  // CHECK-DAG: %[[S0D4:.*]] = tensor.extract %[[S0]][%[[C4]]]
  // CHECK-DAG: %[[S0D5:.*]] = tensor.extract %[[S0]][%[[C5]]]
  // CHECK-DAG: %[[S1D5:.*]] = tensor.extract %[[S1]][%[[C5]]]
  // CHECK-DAG: %[[RES:.*]] = tensor.from_elements %[[S0D0]], %[[C1]], %[[C8]], %[[C4]], %[[S0D4]], %[[S0D5]], %[[S1D5]]
  // CHECK:     return %[[RES]]
  %0 = shape.shape_of %arg0 : tensor<?x1x1x4x?x?x1xf32> -> tensor<7xindex>
  %1 = shape.shape_of %arg1 : tensor<1x8x1x?x1x?xf32> -> tensor<6xindex>
  %2 = shape.broadcast %0, %1 : tensor<7xindex>, tensor<6xindex>
      -> tensor<7xindex>
  func.return %2 : tensor<7xindex>
}

// -----

// CHECK-LABEL: @very_dynamic_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>
func.func @very_dynamic_bcast(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> tensor<1xindex> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[BCASTED:.*]] = shape.broadcast %[[S0]], %[[S1]]
  // CHECK:     return %[[BCASTED]]
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
      -> tensor<1xindex>
  func.return %2 : tensor<1xindex>
}

// -----

// CHECK-LABEL: @broadcast_w_dyn_ty
// CHECK-SAME:  %[[ARG:.*]]: tensor<1xindex>
func.func @broadcast_w_dyn_ty(%arg0: tensor<1xindex>) -> tensor<?xindex>{
  // CHECK: %[[C0:.*]] = arith.constant 0
  // CHECK: %[[D0:.*]] = tensor.extract %[[ARG]][%[[C0]]]
  // CHECK: %[[UNCAST:.*]] = tensor.from_elements %[[D0]]
  // CHECK: %[[CAST:.*]] = tensor.cast %[[UNCAST]] : tensor<1xindex> to tensor<?xindex>
  // CHECK: return %[[CAST]]
   %0 = shape.broadcast %arg0, %arg0
       : tensor<1xindex>, tensor<1xindex> -> tensor<?xindex>
   func.return %0 : tensor<?xindex>
}

// -----

// CHECK-LABEL: @broadcast_scalar_w_dyn_ty
// CHECK-SAME:  %[[ARG:.*]]: tensor<0xindex>
func.func @broadcast_scalar_w_dyn_ty(%arg0: tensor<0xindex>) -> tensor<?xindex>{
  // CHECK: %[[UNCAST:.*]] = arith.constant dense<> : tensor<0xindex>
  // CHECK: %[[CAST:.*]] = tensor.cast %[[UNCAST]] : tensor<0xindex> to tensor<?xindex>
  // CHECK: return %[[CAST]]
   %0 = shape.broadcast %arg0, %arg0
       : tensor<0xindex>, tensor<0xindex> -> tensor<?xindex>
   func.return %0 : tensor<?xindex>
}
