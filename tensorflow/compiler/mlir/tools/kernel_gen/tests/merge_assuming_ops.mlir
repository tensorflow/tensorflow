// RUN: kernel-gen-opt --split-input-file --allow-unregistered-dialect \
// RUN:   --mhlo-merge-assuming-ops --canonicalize --cse %s | \
// RUN: FileCheck %s

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>)
func.func @shape_of_unary(%arg : tensor<?x32xi16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x32xi16> -> tensor<?xindex>
  // CHECK: "use"(%[[SHAPE]])
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%1) : (tensor<?xindex>) -> ()
  func.return
}

// -----

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_nary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>)
func.func @shape_of_nary(%arg0 : tensor<?x32xf16>, %arg1 : tensor<?x32xf16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]] : tensor<?x32xf16> -> tensor<?xindex>
  // CHECK: "use"(%[[SHAPE]])
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %1 = mhlo.subtract %0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %1 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%2) : (tensor<?xindex>) -> ()
  func.return
}

// -----

// CHECK-LABEL: @inline_bcasted_shape_operands
// CHECK-SAME: (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>)
func.func @inline_bcasted_shape_operands(%a : tensor<?xindex>, %b : tensor<?xindex>,
    %c : tensor<?xindex>) -> !shape.witness {
  // CHECK-NOT: shape.broadcast
  // CHECK:     %[[WITNESS:.*]] = shape.cstr_broadcastable %[[A]], %[[B]], %[[C]]
  // CHECK:     return %[[WITNESS]] : !shape.witness
  %0 = shape.broadcast %a, %b : tensor<?xindex>, tensor<?xindex>
      -> tensor<?xindex>
  %1 = shape.cstr_broadcastable %0, %c : tensor<?xindex>, tensor<?xindex>
  func.return %1 : !shape.witness
}

// -----

// CHECK-LABEL: @move_shape_of_into_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<?x32xf32>)
func.func @move_shape_of_into_assuming(%arg0 : !shape.witness,
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
  func.return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_cstr_broadcastable_into_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2xindex>)
func.func @move_cstr_broadcastable_into_assuming(%arg0 : !shape.witness,
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
  func.return %1 : !shape.witness
}

// -----

// CHECK-LABEL: @not_move_shape_of_into_assuming
// CHECK-SAME: (%[[W:.*]]: !shape.witness, %[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>)
func.func @not_move_shape_of_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?x32xf32>, %arg2 : tensor<?x32xf32>) -> tensor<2xindex> {
  // CHECK: %[[S:.*]] = shape.shape_of %[[ARG1]]
  // CHECK: %[[ASS_RES:.*]] = shape.assuming %[[W]]
  // CHECK:   shape.assuming_yield %[[ARG0]]
  // CHECK: }
  // CHECK: "some.other.op"(%[[ASS_RES]])
  // CHECK: return %[[S]]
  %0:2 = shape.assuming %arg0 -> (tensor<?x32xf32>, tensor<?x32xf32>) {
    shape.assuming_yield %arg1, %arg2 : tensor<?x32xf32>, tensor<?x32xf32>
  }
  "some.other.op"(%0#0) : (tensor<?x32xf32>) -> ()
  %2 = shape.shape_of %0#1 : tensor<?x32xf32> -> tensor<2xindex>
  func.return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_cstr_broadcastable_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2xindex>, %[[ARG2:.*]]: tensor<3xindex>)
func.func @move_cstr_broadcastable_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2xindex>, %arg2 : tensor<3xindex>) -> !shape.witness {
  // CHECK:     %[[WITNESS:.*]] = shape.cstr_broadcastable %[[ARG1]], %[[ARG2]]
  // CHECK-NOT: assuming
  // CHECK-NOT: cstr_broadcastable
  // CHECK:     return %[[WITNESS]]
  %0 = shape.assuming %arg0 -> (!shape.witness) {
    %1 = shape.cstr_broadcastable %arg1, %arg2 : tensor<2xindex>, tensor<3xindex>
    shape.assuming_yield %1 : !shape.witness
  }
  func.return %0 : !shape.witness
}

// -----

// CHECK-LABEL: @move_elementwise_into_assuming
// CHECK-SAME:  (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<?xf32>)
func.func @move_elementwise_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:     %[[RES:.*]] = shape.assuming %[[ARG0]]
  // CHECK:       %[[SOME:.*]] = "some.op"
  // CHECK:       %[[TANH:.*]] = mhlo.tanh %[[ARG1]]
  // CHECK:       %[[BCAST_ADD:.*]] = chlo.broadcast_add %[[TANH]], %[[SOME]]
  // CHECK:       shape.assuming_yield %[[BCAST_ADD]]
  // CHECK-NOT: tanh
  // CHECK-NOT: broadcast_add
  // CHECK:     return %[[RES]]
  %0:2 = shape.assuming %arg0 -> (tensor<?xf32>, tensor<?xf32>) {
    %1 = "some.op"() : () -> tensor<?xf32>
    shape.assuming_yield %arg1, %1 : tensor<?xf32>, tensor<?xf32>
  }
  %1 = mhlo.tanh %arg1 : tensor<?xf32>
  %2 = chlo.broadcast_add %1, %0#1
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func.func @move_shape_of_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK:     %[[SHAPE:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-NOT: assuming
  // CHECK-NOT: cstr_broadcastable
  // CHECK:     return %[[SHAPE]]
    %0 = shape.assuming %arg0 -> (tensor<2xindex>) {
    %1 = shape.shape_of %arg1 : tensor<2x?xf32> -> tensor<2xindex>
    shape.assuming_yield %1 : tensor<2xindex>
  }
  func.return %0 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func.func @move_shape_of_out_of_assuming(%arg0 : !shape.witness,
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
  func.return %0#1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @not_move_shape_of_out_of_assuming
// CHECK-SAME: (%[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<2x?xf32>)
func.func @not_move_shape_of_out_of_assuming(%arg0 : !shape.witness,
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
  func.return %0 : tensor<2xindex>
}

// -----

// CHECK: @merge_assuming_ops
// CHECK: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>)
func.func @merge_assuming_ops(%arg0: tensor<?x32xf16>, %arg1 : tensor<?x32xf16>,
    %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK:      %[[SHAPE0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK:      %[[SHAPE1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK:      %[[SHAPE2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK:      %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE0]], %[[SHAPE1]], %[[SHAPE2]]
  // CHECK:      %[[MERGED:.*]]:2 = shape.assuming %[[WITNESS]]
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
  func.return %7 : tensor<?x?x32xf16>
}

// -----

// Do not merge assuming ops if witness will not dominate use.
// CHECK: @do_not_merge_assuming_ops
func.func @do_not_merge_assuming_ops() {
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
  func.return
}

// -----

// CHECK:      @eliminate_extent_tensor_cast
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x?x4xf32>)
func.func @eliminate_extent_tensor_cast(%arg : tensor<2x?x4xf32>) {
  // CHECK-NOT:  shape_of
  // CHECK:      %[[RESULT:.*]] = shape.shape_of %[[ARG]] : tensor<2x?x4xf32> -> tensor<3xindex>
  // CHECK-NEXT: "use"(%[[RESULT]]) : (tensor<3xindex>) -> ()
  %0 = shape.shape_of %arg : tensor<2x?x4xf32> -> tensor<?xindex>
  %1 = tensor.cast %0 : tensor<?xindex> to tensor<3xindex>
  "use"(%1) : (tensor<3xindex>) -> ()
  func.return
}

// -----

// CHECK-LABEL: @redundant_cstr_broadcastable
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xindex>, %[[ARG1:.*]]: tensor<?xindex>)
func.func @redundant_cstr_broadcastable(%arg0: tensor<?xindex>,
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
  func.return
}

// -----

// CHECK-LABEL: @move_assuming_all_over_assuming_region
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xindex>, %[[ARG1:.*]]: tensor<?xindex>, %[[ARG2:.*]]: tensor<?xindex>, %[[ARG3:.*]]: tensor<?xindex>, %[[ARG4:.*]]: tensor<?xindex>
func.func @move_assuming_all_over_assuming_region(%arg0: tensor<?xindex>,
    %arg1 : tensor<?xindex>, %arg2 : tensor<?xindex>, %arg3 : tensor<?xindex>,
    %arg4 : tensor<?xindex>) {
  // CHECK-DAG: %[[CSTR0:.*]] = shape.cstr_broadcastable %[[ARG0]], %[[ARG1]]
  // CHECK-DAG: %[[CSTR1:.*]] = shape.cstr_broadcastable %[[ARG1]], %[[ARG2]]
  // CHECK-DAG: %[[CSTR2:.*]] = shape.cstr_broadcastable %[[ARG2]], %[[ARG3]]
  // CHECK-DAG: %[[CSTR3:.*]] = shape.cstr_broadcastable %[[ARG3]], %[[ARG4]]
  // CHECK-DAG: %[[CSTR_ALL:.*]] = shape.assuming_all %[[CSTR0]], %[[CSTR1]], %[[CSTR2]], %[[CSTR3]]
  // CHECK:     shape.assuming %[[CSTR_ALL]] {
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
  func.return
}

// -----

// CHECK-LABEL: @move_down_into_assuming
// CHECK-SAME:  (%[[ARG:.*]]: tensor<?x32xi16>, %[[W:.*]]: !shape.witness)
func.func @move_down_into_assuming(%arg0: tensor<?x32xi16>, %w: !shape.witness) -> tensor<?x32xf16> {
  // CHECK: %[[RES:.*]] = shape.assuming %[[W]]
  // CHECK:   %[[INNER_RES:.*]] = mhlo.convert %[[ARG]]
  // CHECK:   shape.assuming_yield %[[INNER_RES]]
  // CHECK: }
  // CHECK: return %[[RES]]
  %0 = mhlo.convert %arg0 : (tensor<?x32xi16>) -> tensor<?x32xf16>
  "some.possibly_side_effecting_op"() : () -> ()
  %4 = shape.assuming %w -> (tensor<?x32xf16>) {
    shape.assuming_yield %0 : tensor<?x32xf16>
  }
  func.return %4 : tensor<?x32xf16>
}
