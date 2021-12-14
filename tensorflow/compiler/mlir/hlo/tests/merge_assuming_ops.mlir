// RUN: mlir-hlo-opt --split-input-file --allow-unregistered-dialect \
// RUN:   --mhlo-merge-assuming-ops --canonicalize --cse %s | \
// RUN: FileCheck %s

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>)
func @shape_of_unary(%arg : tensor<?x32xi16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x32xi16> -> tensor<?xindex>
  // CHECK: "use"(%[[SHAPE]])
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%1) : (tensor<?xindex>) -> ()
  return
}

// -----

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_nary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>)
func @shape_of_nary(%arg0 : tensor<?x32xf16>, %arg1 : tensor<?x32xf16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]] : tensor<?x32xf16> -> tensor<?xindex>
  // CHECK: "use"(%[[SHAPE]])
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %1 = mhlo.subtract %0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %1 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%2) : (tensor<?xindex>) -> ()
  return
}

// -----

// Broadcasts can be moved up over unary shape-preserving operations.
// CHECK-LABEL: @bcast_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>, %[[OUT_DIMS:.*]]: tensor<3xindex>)
func @bcast_unary(%arg : tensor<?x32xi16>, %out_dims : tensor<3xindex>)
    -> tensor<?x?x32xf16> {
  // CHECK:      %[[BCASTED_OPERAND:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[OUT_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x32xi16>, tensor<3xindex>) -> tensor<?x?x32xi16>
  // CHECK:      "mhlo.convert"(%[[BCASTED_OPERAND]]) : (tensor<?x?x32xi16>) -> tensor<?x?x32xf16>
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %out_dims) {
      broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> } :
      (tensor<?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
  return %1 : tensor<?x?x32xf16>
}

// -----

// Broadcasts can be moved up over n-ary shape-preserving operations.
// CHECK-LABEL: @bcast_nary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[OUT_DIMS:.*]]: tensor<3xindex>)
func @bcast_nary(%arg0 : tensor<?x32xf32>, %arg1 : tensor<?x32xf32>,
    %out_dims : tensor<3xindex>) -> tensor<?x?x32xf32> {
  // CHECK-NOT: subtract
  // CHECK:     %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[OUT_DIMS]])
  // CHECK:     %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[OUT_DIMS]])
  // CHECK:     %{{.*}} = mhlo.subtract %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : tensor<?x?x32xf32>
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %out_dims) {
      broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> } :
      (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
  return %1 : tensor<?x?x32xf32>
}

// -----

// Exemplary IR as it appears in the lowering with `tf.Sub` and `tf.Cast`.
// CHECK-LABEL: @cast_sub
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xi16>, %[[ARG1:.*]]: tensor<?x?x32xf16>) -> tensor<?x?x32xf16>
func @cast_sub(%arg0: tensor<?x32xi16>, %arg1: tensor<?x?x32xf16>)
    -> tensor<?x?x32xf16> {
  // CHECK-NOT: convert
  // CHECK:     %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %{{.*}})
  // CHECK:     %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %{{.*}})
  // CHECK:     %[[CONVERTED_BCASTED_ARG0:.*]] = "mhlo.convert"(%[[BCASTED_ARG0]]) : (tensor<?x?x32xi16>) -> tensor<?x?x32xf16>
  // CHECK:     %{{.*}} = mhlo.subtract %[[BCASTED_ARG1]], %[[CONVERTED_BCASTED_ARG0]] : tensor<?x?x32xf16>
  %0 = "mhlo.convert"(%arg0) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %arg1 : tensor<?x?x32xf16> -> tensor<?xindex>
  %2 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  %3 = shape.cstr_broadcastable %1, %2 : tensor<?xindex>, tensor<?xindex>
  %4 = shape.assuming %3 -> (tensor<?x?x32xf16>) {
    %5 = shape.shape_of %arg1 : tensor<?x?x32xf16> -> tensor<?xindex>
    %6 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
    %7 = shape.broadcast %5, %6 : tensor<?xindex>, tensor<?xindex>
        -> tensor<?xindex>
    %8 = tensor.cast %7 : tensor<?xindex> to tensor<3xindex>
    %9 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %8) {
        broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} :
        (tensor<?x?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %10 = "mhlo.dynamic_broadcast_in_dim"(%0, %8) {
        broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
        (tensor<?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %11 = mhlo.subtract %9, %10 : tensor<?x?x32xf16>
    shape.assuming_yield %11 : tensor<?x?x32xf16>
  }
  return %4 : tensor<?x?x32xf16>
}

// -----

// CHECK-LABEL: @inline_bcasted_shape_operands
// CHECK-SAME: (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>)
func @inline_bcasted_shape_operands(%a : tensor<?xindex>, %b : tensor<?xindex>,
    %c : tensor<?xindex>) -> !shape.witness {
  // CHECK-NOT: shape.broadcast
  // CHECK:     %[[WITNESS:.*]] = shape.cstr_broadcastable %[[A]], %[[B]], %[[C]]
  // CHECK:     return %[[WITNESS]] : !shape.witness
  %0 = shape.broadcast %a, %b : tensor<?xindex>, tensor<?xindex>
      -> tensor<?xindex>
  %1 = shape.cstr_broadcastable %0, %c : tensor<?xindex>, tensor<?xindex>
  return %1 : !shape.witness
}

// -----

// CHECK-LABEL: @move_shape_of_into_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<?x32xf32>)
func @move_shape_of_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?x32xf32>) -> tensor<2xindex> {
  // CHECK:     %[[ASSUMING_RESULTS:.*]]:3 = shape.assuming %[[ARG0]] -> (tensor<?x32xf32>, tensor<?x32xf32>, tensor<2xindex>) {
  // CHECK:       %[[DUMMY_TENSOR:.*]] = "dummy.tensor"() : () -> tensor<?x32xf32>
  // CHECK:       %[[SHAPE:.*]] = shape.shape_of %[[DUMMY_TENSOR]]
  // CHECK:       shape.assuming_yield %[[ARG1]], %[[DUMMY_TENSOR]], %[[SHAPE]]
  // CHECK:     }
  // CHECK-NOT: shape_of
  // CHECK:     return %[[ASSUMING_RESULTS]]#2
  %0:2 = shape.assuming %arg0 -> (tensor<?x32xf32>, tensor<?x32xf32>) {
    %1 = "dummy.tensor"() : () -> tensor<?x32xf32>
    shape.assuming_yield %arg1, %1 : tensor<?x32xf32>, tensor<?x32xf32>
  }
  %2 = shape.shape_of %0#1 : tensor<?x32xf32> -> tensor<2xindex>
  "use"(%0#0, %0#1) : (tensor<?x32xf32>, tensor<?x32xf32>) -> ()
  return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_cstr_broadcastable_into_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2xindex>)
func @move_cstr_broadcastable_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2xindex>) -> !shape.witness {
  // CHECK:     %[[ASSUMING_RESULTS:.*]]:3 = shape.assuming %[[ARG0]] -> (tensor<2xindex>, tensor<3xindex>, !shape.witness) {
  // CHECK:       %[[DUMMY_TENSOR:.*]] = "dummy.tensor"() : () -> tensor<3xindex>
  // CHECK:       %[[WITNESS:.*]] = shape.cstr_broadcastable %[[ARG1]], %[[DUMMY_TENSOR]]
  // CHECK:       shape.assuming_yield %[[ARG1]], %[[DUMMY_TENSOR]], %[[WITNESS]]
  // CHECK:     }
  // CHECK-NOT: cstr_broadcastable
  // CHECK:     return %[[ASSUMING_RESULTS]]#2
  %0:2 = shape.assuming %arg0 -> (tensor<2xindex>, tensor<3xindex>) {
    %1 = "dummy.tensor"() : () -> tensor<3xindex>
    shape.assuming_yield %arg1, %1 : tensor<2xindex>, tensor<3xindex>
  }
  %1 = shape.cstr_broadcastable %arg1, %0#1 : tensor<2xindex>, tensor<3xindex>
  "use"(%0#0, %0#1) : (tensor<2xindex>, tensor<3xindex>) -> ()
  return %1 : !shape.witness
}

// -----

// CHECK-LABEL: @not_move_shape_of_into_assuming
func @not_move_shape_of_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?x32xf32>, %arg2 : tensor<?x32xf32>) -> tensor<2xindex> {
  // CHECK:      shape.assuming
  // CHECK-SAME: {
  // CHECK-NOT:    shape_of
  // CHECK:      }
  // CHECK:     "some.other.op"
  // CHECK:     shape_of
  %0:2 = shape.assuming %arg0 -> (tensor<?x32xf32>, tensor<?x32xf32>) {
    shape.assuming_yield %arg1, %arg2 : tensor<?x32xf32>, tensor<?x32xf32>
  }
  "some.other.op"() : () -> ()
  %2 = shape.shape_of %0#1 : tensor<?x32xf32> -> tensor<2xindex>
  return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_cstr_broadcastable_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2xindex>, %[[ARG2:.*]]: tensor<3xindex>)
func @move_cstr_broadcastable_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2xindex>, %arg2 : tensor<3xindex>) -> !shape.witness {
  // CHECK:     %[[WITNESS:.*]] = shape.cstr_broadcastable %[[ARG1]], %[[ARG2]]
  // CHECK-NOT: assuming
  // CHECK-NOT: cstr_broadcastable
  // CHECK:     return %[[WITNESS]]
  %0 = shape.assuming %arg0 -> (!shape.witness) {
    %1 = shape.cstr_broadcastable %arg1, %arg2 : tensor<2xindex>, tensor<3xindex>
    shape.assuming_yield %1 : !shape.witness
  }
  return %0 : !shape.witness
}

// -----

// CHECK-LABEL: @move_elementwise_into_assuming
// CHECK-SAME:  (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<?xf32>)
func @move_elementwise_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:     %[[RES:.*]] = shape.assuming %[[ARG0]]
  // CHECK:       %[[SOME:.*]] = "some.op"
  // CHECK:       %[[TANH:.*]] = "mhlo.tanh"(%[[ARG1]])
  // CHECK:       %[[BCAST_ADD:.*]] = chlo.broadcast_add %[[TANH]], %[[SOME]]
  // CHECK:       shape.assuming_yield %[[BCAST_ADD]]
  // CHECK-NOT: tanh
  // CHECK-NOT: broadcast_add
  // CHECK:     return %[[RES]]
  %0:2 = shape.assuming %arg0 -> (tensor<?xf32>, tensor<?xf32>) {
    %1 = "some.op"() : () -> tensor<?xf32>
    shape.assuming_yield %arg1, %1 : tensor<?xf32>, tensor<?xf32>
  }
  %1 = "mhlo.tanh"(%arg1) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = chlo.broadcast_add %1, %0#1
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func @move_shape_of_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK:     %[[SHAPE:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-NOT: assuming
  // CHECK-NOT: cstr_broadcastable
  // CHECK:     return %[[SHAPE]]
    %0 = shape.assuming %arg0 -> (tensor<2xindex>) {
    %1 = shape.shape_of %arg1 : tensor<2x?xf32> -> tensor<2xindex>
    shape.assuming_yield %1 : tensor<2xindex>
  }
  return %0 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func @move_shape_of_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK:     %[[SHAPE:.*]] = shape.shape_of %[[ARG1]]
  // CHECK:     %{{.*}} = shape.assuming %[[ARG0]] -> (tensor<2x?xf32>) {
  // CHECK:       %[[SOME_VAL:.*]] = "some.op"() : () -> tensor<2x?xf32>
  // CHECK:       shape.assuming_yield %[[SOME_VAL]] : tensor<2x?xf32>
  // CHECK:     }
  // CHECK:     return %[[SHAPE]]
  %0:2 = shape.assuming %arg0 -> (tensor<2x?xf32>, tensor<2xindex>) {
    %1 = "some.op"() : () -> (tensor<2x?xf32>)
    %2 = shape.shape_of %arg1 : tensor<2x?xf32> -> tensor<2xindex>
    shape.assuming_yield %1, %2 : tensor<2x?xf32>, tensor<2xindex>
  }
  "use"(%0#0, %0#1) : (tensor<2x?xf32>, tensor<2xindex>) -> ()
  return %0#1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @not_move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func @not_move_shape_of_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK-NOT:  shape_of
  // CHECK:      shape.assuming
  // CHECK-SAME: {
  // CHECK:        "some.tensor"
  // CHECK:        shape_of
  // CHECK:      }
  %0 = shape.assuming %arg0 -> (tensor<2xindex>) {
    %1 = "some.tensor"() : () -> tensor<2x?xf32>
    %2 = shape.shape_of %1 : tensor<2x?xf32> -> tensor<2xindex>
    shape.assuming_yield %2 : tensor<2xindex>
  }
  return %0 : tensor<2xindex>
}

// -----

// CHECK: @merge_assuming_ops
// CHECK: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>)
func @merge_assuming_ops(%arg0: tensor<?x32xf16>, %arg1 : tensor<?x32xf16>,
    %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK:      %[[SHAPE0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK:      %[[SHAPE1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK:      %[[SHAPE2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK:      %[[WITNESS0:.*]] = shape.cstr_broadcastable %[[SHAPE0]], %[[SHAPE1]]
  // CHECK:      %[[WITNESS1:.*]] = shape.cstr_broadcastable %[[SHAPE0]], %[[SHAPE1]], %[[SHAPE2]]
  // CHECK:      %[[COMBINED_WITNESS:.*]] = shape.assuming_all %[[WITNESS0]], %[[WITNESS1]]
  // CHECK:      %[[MERGED:.*]]:2 = shape.assuming %[[COMBINED_WITNESS]]
  // CHECK-SAME: {
  // CHECK:        "some.op"
  // CHECK:        %[[RESULT0:.*]] = "some.producer"
  // CHECK:        "another.op"
  // CHECK:        %[[RESULT1:.*]] = "another.producer"
  // CHECK:        shape.assuming_yield %[[RESULT0]], %[[RESULT1]]
  // CHECK:      }
  // CHECK:      return %[[MERGED]]#1
  %0 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
  %2 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
  %3 = shape.cstr_broadcastable %0, %1 : tensor<2xindex>, tensor<2xindex>
  %4 = shape.cstr_broadcastable %0, %1, %2 : tensor<2xindex>, tensor<2xindex>,
      tensor<3xindex>
  %5 = shape.assuming %3 -> (tensor<?x32xf16>) {
    "some.op"() : () -> ()
    %6 = "some.producer"() : () -> tensor<?x32xf16>
    shape.assuming_yield %6 : tensor<?x32xf16>
  }
  %7 = shape.assuming %4 -> (tensor<?x?x32xf16>) {
    "another.op"() : () -> ()
    %8 = "another.producer"() : () -> tensor<?x?x32xf16>
    shape.assuming_yield %8 : tensor<?x?x32xf16>
  }
  "use"(%5, %7) : (tensor<?x32xf16>, tensor<?x?x32xf16>) -> ()
  return %7 : tensor<?x?x32xf16>
}

// -----

// Do not merge assuming ops if witness will not dominate use.
// CHECK: @do_not_merge_assuming_ops
func @do_not_merge_assuming_ops() {
  // CHECK: shape.assuming
  // CHECK: shape.assuming
  %0 = "some.witness"() : () -> !shape.witness
  %1 = shape.assuming %0 -> (!shape.witness) {
    %2 = "some.witness"() : () -> !shape.witness
    shape.assuming_yield %2 : !shape.witness
  }
  shape.assuming %1 {
    "some.op"() : () -> ()
    shape.assuming_yield
  }
  return
}

// -----

// CHECK:      @eliminate_extent_tensor_cast
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x?x4xf32>)
func @eliminate_extent_tensor_cast(%arg : tensor<2x?x4xf32>) {
  // CHECK-NOT:  shape_of
  // CHECK:      %[[RESULT:.*]] = shape.shape_of %[[ARG]] : tensor<2x?x4xf32> -> tensor<3xindex>
  // CHECK-NEXT: "use"(%[[RESULT]]) : (tensor<3xindex>) -> ()
  %0 = shape.shape_of %arg : tensor<2x?x4xf32> -> tensor<?xindex>
  %1 = tensor.cast %0 : tensor<?xindex> to tensor<3xindex>
  "use"(%1) : (tensor<3xindex>) -> ()
  return
}

// -----

// Exemplary IR as it appears in the lowering of two subsequent `tf.Sub` ops.
// CHECK-LABEL: @sub_sub
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>)
func @sub_sub(%arg0: tensor<?x32xf16>, %arg1 : tensor<?x32xf16>,
    %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK-DAG:  %[[SHAPE0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:  %[[SHAPE1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG:  %[[SHAPE2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG:  %[[WITNESS0:.*]] = shape.cstr_broadcastable %[[SHAPE0]], %[[SHAPE1]]
  // CHECK-DAG:  %[[WITNESS1:.*]] = shape.cstr_broadcastable %[[SHAPE2]], %[[SHAPE0]], %[[SHAPE1]]
  // CHECK-DAG:  %[[COMBINED_WITNESS:.*]] = shape.assuming_all %[[WITNESS0]], %[[WITNESS1]]
  // CHECK:      %[[ASSUMING_RESULT:.*]] = shape.assuming %[[COMBINED_WITNESS]]
  // CHECK:        %[[BCASTED_SHAPE01:.*]] = shape.broadcast %[[SHAPE0]], %[[SHAPE1]]
  // CHECK:        %[[BCASTED_SHAPE012:.*]] = shape.broadcast %[[SHAPE2]], %[[BCASTED_SHAPE01]]
  // CHECK:        %[[BCASTED_ARG2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[BCASTED_SHAPE012]]) {broadcast_dimensions = dense<[0, 1, 2]>
  // CHECK:        %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[BCASTED_SHAPE012]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
  // CHECK:        %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[BCASTED_SHAPE012]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
  // CHECK:        %[[TMP:.*]] = mhlo.subtract %[[BCASTED_ARG0]], %[[BCASTED_ARG1]]
  // CHECK:        %[[RESULT:.*]] = mhlo.subtract %[[BCASTED_ARG2]], %[[TMP]]
  // CHECK:        shape.assuming_yield %[[RESULT]]
  // CHECK:      return %[[ASSUMING_RESULT]]
  %0 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<2xindex>, tensor<2xindex>
  %3 = shape.assuming %2 -> (tensor<?x32xf16>) {
    %8 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
    %9 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
    %10 = shape.broadcast %8, %9 : tensor<2xindex>, tensor<2xindex> -> tensor<?xindex>
    %11 = tensor.cast %10 : tensor<?xindex> to tensor<2xindex>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %11) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x32xf16>, tensor<2xindex>) -> tensor<?x32xf16>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %11) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x32xf16>, tensor<2xindex>) -> tensor<?x32xf16>
    %14 = mhlo.subtract %12, %13 : tensor<?x32xf16>
    shape.assuming_yield %14 : tensor<?x32xf16>
  }
  %4 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
  %5 = shape.shape_of %3 : tensor<?x32xf16> -> tensor<2xindex>
  %6 = shape.cstr_broadcastable %4, %5 : tensor<3xindex>, tensor<2xindex>
  %7 = shape.assuming %6 -> (tensor<?x?x32xf16>) {
    %8 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
    %9 = shape.shape_of %3 : tensor<?x32xf16> -> tensor<2xindex>
    %10 = shape.broadcast %8, %9 : tensor<3xindex>, tensor<2xindex> -> tensor<?xindex>
    %11 = tensor.cast %10 : tensor<?xindex> to tensor<3xindex>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %11) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%3, %11) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %14 = mhlo.subtract %12, %13 : tensor<?x?x32xf16>
    shape.assuming_yield %14 : tensor<?x?x32xf16>
  }
  return %7 : tensor<?x?x32xf16>
}

// -----

// CHECK-LABEL: @redundant_cstr_broadcastable
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xindex>, %[[ARG1:.*]]: tensor<?xindex>)
func @redundant_cstr_broadcastable(%arg0: tensor<?xindex>,
    %arg1 : tensor<?xindex>) {
  // CHECK-DAG:  %[[WITNESS:.*]] = shape.cstr_broadcastable %[[ARG0]], %[[ARG1]]
  // CHECK:      shape.assuming %[[WITNESS]]
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  %1 = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  %2 = shape.assuming_all %0, %1
  shape.assuming %2 -> () {
    "some.op"() : () -> ()
    shape.assuming_yield
  }
  return
}

// -----

// CHECK-LABEL: @move_assuming_all_over_assuming_region
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xindex>, %[[ARG1:.*]]: tensor<?xindex>, %[[ARG2:.*]]: tensor<?xindex>, %[[ARG3:.*]]: tensor<?xindex>, %[[ARG4:.*]]: tensor<?xindex>
func @move_assuming_all_over_assuming_region(%arg0: tensor<?xindex>,
    %arg1 : tensor<?xindex>, %arg2 : tensor<?xindex>, %arg3 : tensor<?xindex>,
    %arg4 : tensor<?xindex>) {
  // CHECK-DAG: %[[CSTR0:.*]] = shape.cstr_broadcastable %[[ARG0]], %[[ARG1]]
  // CHECK-DAG: %[[CSTR1:.*]] = shape.cstr_broadcastable %[[ARG1]], %[[ARG2]]
  // CHECK-DAG: %[[CSTR_ALL01:.*]] = shape.assuming_all %[[CSTR0]], %[[CSTR1]]
  // CHECK-DAG: %[[CSTR2:.*]] = shape.cstr_broadcastable %[[ARG2]], %[[ARG3]]
  // CHECK-DAG: %[[CSTR3:.*]] = shape.cstr_broadcastable %[[ARG3]], %[[ARG4]]
  // CHECK-DAG: %[[CSTR_ALL23:.*]] = shape.assuming_all %[[CSTR2]], %[[CSTR3]]
  // CHECK-DAG: %[[CSTR_ALL0123:.*]] = shape.assuming_all %[[CSTR_ALL01]], %[[CSTR_ALL23]]
  // CHECK:     shape.assuming %[[CSTR_ALL0123]] {
  // CHECK:       "some.op"()
  // CHECK:       "some.op"()
  // CHECK:     }
  // CHECK:     return
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  %1 = shape.cstr_broadcastable %arg1, %arg2 : tensor<?xindex>, tensor<?xindex>
  %2 = shape.assuming_all %0, %1
  shape.assuming %2 -> () {
    "some.op"() : () -> ()
  }
  %3 = shape.cstr_broadcastable %arg2, %arg3 : tensor<?xindex>, tensor<?xindex>
  %4 = shape.cstr_broadcastable %arg3, %arg4 : tensor<?xindex>, tensor<?xindex>
  %5 = shape.assuming_all %3, %4
  shape.assuming %5 -> () {
    "some.op"() : () -> ()
  }
  return
}

// -----

// CHECK-LABEL: @bcast_select_scalar_pred
// CHECK-SAME:  %[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<2xindex>
func @bcast_select_scalar_pred(%pred : tensor<i1>, %arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %shape : tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK:      %[[BCASTED_PRED:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[PRED]], %[[SHAPE]])
  // CHECK-SAME:   broadcast_dimensions = dense<>
  // CHECK:      %[[BCASTED_LHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[LHS]], %[[SHAPE]])
  // CHECK-SAME:   broadcast_dimensions = dense<[0, 1]>
  // CHECK:      %[[BCASTED_RHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[RHS]], %[[SHAPE]])
  // CHECK-SAME:   broadcast_dimensions = dense<[0, 1]>
  // CHECK:      %[[RESULT:.*]] = "mhlo.select"(%[[BCASTED_PRED]], %[[BCASTED_LHS]], %[[BCASTED_RHS]])
  // CHECK:      return %[[RESULT]]
  %0 = "mhlo.select"(%pred, %arg0, %arg1)
      : (tensor<i1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape)
      { broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> }
      : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
