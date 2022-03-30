// RUN: tf-tfrt-opt -verify-diagnostics -tf-jitrt-test-clustering-policy %s    \
// RUN:   | FileCheck %s

// -------------------------------------------------------------------------- //
// tf._FusedMatMul
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @fused_matmul_rank_constraint
func.func @fused_matmul_rank_constraint(%arg0 : tensor<?x?xf32>,
                                   %arg1 : tensor<?x?xf32>,
                                   %arg2 : tensor<?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  // expected-remark@below {{operand #2 constrained to: rank}}
  %0 = "tf._FusedMatMul"(%arg0, %arg1, %arg2) {fused_ops = ["BiasAdd"]}
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @fused_matmul_unknown_fusion
func.func @fused_matmul_unknown_fusion(%arg0 : tensor<?x?xf32>,
                                  %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf._FusedMatMul"(%arg0, %arg1) {fused_ops = ["NotAFusion"]}
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.AddV2 (as an example of Cwise Binary Operation)
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @add_no_constraint
func.func @add_no_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  %0 = "tf.AddV2"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @add_rank_constraint
func.func @add_rank_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  %0 = "tf.AddV2"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @add_shape_constraint
func.func @add_shape_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.AddV2"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @add_value_constraint
func.func @add_value_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.AddV2"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.BatchMatMulV2
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @batchmatmulv2_no_constraint
func.func @batchmatmulv2_no_constraint(%arg0 : tensor<?x?x?xf32>,
                                  %arg1 : tensor<?x?x?xf32>)
    -> tensor<?x?x?xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false}
       : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @batchmatmulv2_rank_constraint
func.func @batchmatmulv2_rank_constraint(%arg0 : tensor<?x?x?xf32>,
                                    %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xf32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false}
       : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @batchmatmulv2_shape_constraint
func.func @batchmatmulv2_shape_constraint(%arg0 : tensor<?x?x?xf32>,
                                     %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xf32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false}
       : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @batchmatmulv2_value_constraint
func.func @batchmatmulv2_value_constraint(%arg0 : tensor<?x?x?xf32>,
                                     %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xf32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false}
       : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Log1p (as an example of Cwise Unary Operation)
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @log1p_no_constraint
func.func @log1p_no_constraint(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Log1p"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @log1p_rank_constraint
func.func @log1p_rank_constraint(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Log1p"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @log1p_shape_constraint
func.func @log1p_shape_constraint(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  %0 = "tf.Log1p"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @log1p_value_constraint
func.func @log1p_value_constraint(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Log1p"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.MatMul
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @matmul_no_constraint
func.func @matmul_no_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = "tf.MatMul"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_rank_constraint
func.func @matmul_rank_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  %0 = "tf.MatMul"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_shape_constraint
func.func @matmul_shape_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.MatMul"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_value_constraint
func.func @matmul_value_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.MatMul"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Mean (as an example of Reduction Operation)
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @mean_no_constraint
func.func @mean_no_constraint(%arg0 : tensor<?x?xf32>,
                         %arg1 : tensor<i32>) -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Mean"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<i32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @mean_rank_constraint
func.func @mean_rank_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Mean"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @mean_shape_constraint
func.func @mean_shape_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Mean"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @mean_value_constraint
func.func @mean_value_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Mean"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Pack
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @pack_rank_constraints
func.func @pack_rank_constraints(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> (tensor<?x2xf32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  %0 = "tf.Pack"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xf32>) -> tensor<?x2xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: func @pack_shape_constraints
func.func @pack_shape_constraints(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> (tensor<?x2xf32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.Pack"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xf32>) -> tensor<?x2xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: func @pack_value_constraints
func.func @pack_value_constraints(%arg0 : tensor<i32>, %arg1 : tensor<i32>)
    -> (tensor<2xi32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Pack"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<2xi32>
}

// -------------------------------------------------------------------------- //
// tf.Reshape
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @reshape_rank_constraints
func.func @reshape_rank_constraints(%arg0 : tensor<?x?xf32>, %arg1 : tensor<*xi32>)
    -> (tensor<*xf32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.Reshape"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<*xi32>) -> tensor<*xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @reshape_shape_constraints
func.func @reshape_shape_constraints(%arg0 : tensor<?x?xf32>, %arg1 : tensor<*xi32>)
    -> (tensor<*xf32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Reshape"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<*xi32>) -> tensor<*xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<*xf32>
}

// -------------------------------------------------------------------------- //
// tf.Transpose
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @transpose_no_constraint
func.func @transpose_no_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Transpose"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @transpose_rank_constraint
func.func @transpose_rank_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Transpose"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @transpose_shape_constraint
func.func @transpose_shape_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Transpose"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @transpose_value_constraint
func.func @transpose_value_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-remark@below {{operand #0 constrained to: value}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.Transpose"(%arg0, %arg1)
       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Shape
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @shape_no_constraint
func.func @shape_no_constraint(%arg0 : tensor<?x?xf32>) -> tensor<2xi32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: func @shape_shape_constraints
func.func @shape_shape_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<2xi32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: func @shape_value_constraints
func.func @shape_value_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<2xi32> { tf.constraint = "value" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<2xi32>
}

// -------------------------------------------------------------------------- //
// tf.Slice
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @slice_no_constraint
func.func @slice_no_constraint(%arg0 : tensor<?x?xf32>, %arg1: tensor<2xi32>,
                          %arg2: tensor<2xi32>)
    -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  // expected-remark@below {{operand #2 constrained to: value}}
  %0 = "tf.Slice"(%arg0, %arg1, %arg2)
       : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @slice_value_constraint
func.func @slice_value_constraint(%arg0 : tensor<?x?xf32>, %arg1: tensor<2xi32>,
                            %arg2: tensor<2xi32>)
    -> (tensor<?x?xf32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Slice"(%arg0, %arg1, %arg2)
       : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.StridedSlice
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @strided_slice
func.func @strided_slice(%arg0 : tensor<?x?xf32>,
                    %arg1 : tensor<2xi32>) -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  // expected-remark@below {{operand #2 constrained to: value}}
  // expected-remark@below {{operand #3 constrained to: value}}
  %0 = "tf.StridedSlice"(%arg0, %arg1, %arg1, %arg1)
       { begin_mask = 0 : i64,
         ellipsis_mask = 0 : i64,
         end_mask = 0 : i64,
         new_axis_mask = 0 : i64,
         shrink_axis_mask = 0 : i64
       } : (tensor<?x?xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
         -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @strided_slice_of_shape
func.func @strided_slice_of_shape(%arg0 : tensor<?x?xf32>,
                             %arg1 : tensor<1xi32>) -> tensor<1xi32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  // expected-remark@below {{operand #2 constrained to: value}}
  // expected-remark@below {{operand #3 constrained to: value}}
  %1 = "tf.StridedSlice"(%0, %arg1, %arg1, %arg1)
       { begin_mask = 0 : i64,
         ellipsis_mask = 0 : i64,
         end_mask = 0 : i64,
         new_axis_mask = 0 : i64,
         shrink_axis_mask = 1 : i64
       } : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
         -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// -------------------------------------------------------------------------- //
// tf.ConcatV2
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @concatv2_no_constraint
func.func @concatv2_no_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
               %arg2 : tensor<?x?xf32>, %arg3 : tensor<i64>)
    -> tensor<?x?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  // expected-remark@below {{operand #2 constrained to: rank}}
  // expected-remark@below {{operand #3 constrained to: value}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3)
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<i64>)
       -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @concatv2_rank_constraint
func.func @concatv2_rank_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
                               %arg2 : tensor<?x?xf32>, %arg3 : tensor<i64>)
    -> (tensor<?x?xf32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: rank}}
  // expected-remark@below {{operand #2 constrained to: rank}}
  // expected-remark@below {{operand #3 constrained to: value}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3)
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<i64>)
       -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @concatv2_shape_constraint
func.func @concatv2_shape_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
                                %arg2 : tensor<?x?xf32>, %arg3 : tensor<i64>)
    -> (tensor<?x?xf32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  // expected-remark@below {{operand #2 constrained to: shape}}
  // expected-remark@below {{operand #3 constrained to: value}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3)
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<i64>)
       -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @concatv2_value_constraint
func.func @concatv2_value_constraint(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
                                %arg2 : tensor<?x?xf32>, %arg3 : tensor<i64>)
    -> (tensor<?x?xf32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3)
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<i64>)
       -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Fill
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @fill_no_constraint
func.func @fill_no_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<f32>)
    -> tensor<*xf32> {
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @fill_rank_constraint
func.func @fill_rank_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<f32>)
    -> (tensor<*xf32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @fill_shape_constraint
func.func @fill_shape_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<f32>)
    -> (tensor<*xf32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: value}}
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @fill_value_constraint
func.func @fill_value_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<f32>)
    -> (tensor<*xf32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<*xf32>
}

// -------------------------------------------------------------------------- //
// tf.OneHot
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @onehot_no_constraint
func.func @onehot_no_constraint(%arg0: tensor<?xi32>, %arg1: tensor<i32>,
                           %arg2: tensor<f32>, %arg3: tensor<f32>)
    -> tensor<?x2xf32> {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.OneHot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i64}
       : (tensor<?xi32>, tensor<i32>, tensor<f32>, tensor<f32>)
       -> tensor<?x2xf32>
  func.return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: func @onehot_value_constraint
func.func @onehot_value_constraint(%arg0: tensor<?xi32>, %arg1: tensor<i32>,
                              %arg2: tensor<f32>, %arg3: tensor<f32>)
    -> (tensor<?x2xf32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.OneHot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i64}
       : (tensor<?xi32>, tensor<i32>, tensor<f32>, tensor<f32>)
       -> tensor<?x2xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x2xf32>
}

// -------------------------------------------------------------------------- //
// tf.Range
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @range_no_constraint
func.func @range_no_constraint(%arg0 : tensor<i64>, %arg1 : tensor<i64>,
                          %arg2 : tensor<i64>)
    -> tensor<?xi64> {
  %0 = "tf.Range"(%arg0, %arg1, %arg2)
       : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  func.return %0 : tensor<?xi64>
}

// CHECK-LABEL: func @range_rank_constraint
func.func @range_rank_constraint(%arg0 : tensor<i64>, %arg1 : tensor<i64>,
                            %arg2 : tensor<i64>)
    -> (tensor<?xi64> { tf.constraint = "rank" }) {
  %0 = "tf.Range"(%arg0, %arg1, %arg2)
       : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?xi64>
}

// CHECK-LABEL: func @range_shape_constraint
func.func @range_shape_constraint(%arg0 : tensor<i64>, %arg1 : tensor<i64>,
                             %arg2 : tensor<i64>)
    -> (tensor<?xi64> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: value}}
  // expected-remark@below {{operand #1 constrained to: value}}
  // expected-remark@below {{operand #2 constrained to: value}}
  %0 = "tf.Range"(%arg0, %arg1, %arg2)
       : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?xi64>
}

// CHECK-LABEL: func @range_value_constraint
func.func @range_value_constraint(%arg0 : tensor<i64>, %arg1 : tensor<i64>,
                             %arg2 : tensor<i64>)
    -> (tensor<?xi64> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Range"(%arg0, %arg1, %arg2)
       : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?xi64>
}

// -------------------------------------------------------------------------- //
// tf.ExpandDims
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @expand_dims_no_constraint
func.func @expand_dims_no_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>)
    -> tensor<?x1xi32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.ExpandDims"(%arg0, %arg1)
       : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  func.return %0 : tensor<?x1xi32>
}

// CHECK-LABEL: func @expand_dims_rank_constraint
func.func @expand_dims_rank_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>)
    -> (tensor<?x1xi32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.ExpandDims"(%arg0, %arg1)
       : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?x1xi32>
}

// CHECK-LABEL: func @expand_dims_shape_constraint
func.func @expand_dims_shape_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>)
    -> (tensor<?x1xi32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.ExpandDims"(%arg0, %arg1)
       : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?x1xi32>
}

// CHECK-LABEL: func @expand_dims_value_constraint
func.func @expand_dims_value_constraint(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>)
    -> (tensor<?x1xi32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.ExpandDims"(%arg0, %arg1)
       : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x1xi32>
}

// -------------------------------------------------------------------------- //
// tf.BroadcastTo
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @broadcast_to_no_constraint
func.func @broadcast_to_no_constraint(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>)
    -> tensor<?xf32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.BroadcastTo"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @broadcast_to_rank_constraint
func.func @broadcast_to_rank_constraint(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>)
    -> (tensor<?xf32> { tf.constraint = "rank"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: shape}}
  %0 = "tf.BroadcastTo"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @broadcast_to_shape_constraint
func.func @broadcast_to_shape_constraint(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>)
    -> (tensor<?xf32> { tf.constraint = "shape"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  // expected-remark@below {{operand #1 constrained to: value}}
  %0 = "tf.BroadcastTo"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @broadcast_to_value_constraint
func.func @broadcast_to_value_constraint(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>)
    -> (tensor<?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.BroadcastTo"(%arg0, %arg1)
       : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?xf32>
}

// -------------------------------------------------------------------------- //
// tf.Squeeze
// -------------------------------------------------------------------------- //

// CHECK-LABEL: func @squeeze_no_constraint_no_squeeze_dims
func.func @squeeze_no_constraint_no_squeeze_dims(%arg0 : tensor<?x?xi32>)
    -> tensor<?xi32> {
  // expected-remark@below {{operand #0 constrained to: shape}}
  %0 = "tf.Squeeze"(%arg0) : (tensor<?x?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @squeeze_no_constraint
func.func @squeeze_no_constraint(%arg0 : tensor<?x?xi32>)
    -> tensor<?xi32> {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Squeeze"(%arg0) { squeeze_dims = [0] }
       : (tensor<?x?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @squeeze_rank_constraint
func.func @squeeze_rank_constraint(%arg0 : tensor<?x?xi32>)
    -> (tensor<?xi32> { tf.constraint = "rank" }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "tf.Squeeze"(%arg0) { squeeze_dims = [0] }
       : (tensor<?x?xi32>) -> tensor<?xi32>
  // expected-remark@below {{operand #0 constrained to: rank}}
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @squeeze_shape_constraint
func.func @squeeze_shape_constraint(%arg0 : tensor<?x?xi32>)
    -> (tensor<?xi32> { tf.constraint = "shape" }) {
  // expected-remark@below {{operand #0 constrained to: shape}}
  %0 = "tf.Squeeze"(%arg0) { squeeze_dims = [0] }
       : (tensor<?x?xi32>) -> tensor<?xi32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @squeeze_value_constraint
func.func @squeeze_value_constraint(%arg0 : tensor<?x?xi32>)
    -> (tensor<?xi32> { tf.constraint = "value" }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "tf.Squeeze"(%arg0) { squeeze_dims = [0] }
       : (tensor<?x?xi32>) -> tensor<?xi32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?xi32>
}