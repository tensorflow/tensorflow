// RUN: tf-opt %s -tf-replicate-tensor-list-init-ops -verify-diagnostics | FileCheck %s

// CHECK: while_region_op
func.func @while_region_op() {
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: TensorListReserve
  // CHECK: TensorListReserve
  %tl = "tf.TensorListReserve"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %while:1 = "tf.WhileRegion"(%tl) ({
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
      %cond = "tf.false"():()-> tensor<i1>
      "tf.Yield"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
    "tf.Yield"(%barg1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>) -> ()
  }) {is_stateless = false} : (tensor<!tf_type.variant<tensor<?x1xf32>>>) -> (tensor<!tf_type.variant<tensor<?x1xf32>>>)
  func.return
}

// CHECK: while_region_op_empty_tensor_list
func.func @while_region_op_empty_tensor_list() {
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: EmptyTensorList
  // CHECK: EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %while:1 = "tf.WhileRegion"(%tl) ({
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
      %cond = "tf.false"():()-> tensor<i1>
      "tf.Yield"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
    "tf.Yield"(%barg1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>) -> ()
  }) {is_stateless = false} : (tensor<!tf_type.variant<tensor<?x1xf32>>>) -> (tensor<!tf_type.variant<tensor<?x1xf32>>>)
  func.return
}

// CHECK: while_region_op_twosepargs
func.func @while_region_op_twosepargs() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[RESULT0:.*]] = "tf.TensorListReserve"
  // CHECK: %[[RESULT1:.*]] = "tf.TensorListReserve"
  // CHECK: %[[RESULT2:.*]] = "tf.TensorListReserve"
  // CHECK: tf.WhileRegion
  // CHECK-SAME: (%[[RESULT1]], %[[RESULT0]])
  %tl = "tf.TensorListReserve"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %while:2 = "tf.WhileRegion"(%tl, %tl) ({
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>, %barg2: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
      %cond = "tf.false"():()-> tensor<i1>
      "tf.Yield"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>, %barg2: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
    "tf.Yield"(%barg1, %barg2) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>) -> ()
  }) {is_stateless = false} : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>) -> (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>)
  func.return
}

// CHECK: while_region_op_two_sep_args_empty_tensor_list
func.func @while_region_op_two_sep_args_empty_tensor_list() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[RESULT0:.*]] = "tf.EmptyTensorList"
  // CHECK: %[[RESULT1:.*]] = "tf.EmptyTensorList"
  // CHECK: %[[RESULT2:.*]] = "tf.EmptyTensorList"
  // CHECK: tf.WhileRegion
  // CHECK-SAME: (%[[RESULT1]], %[[RESULT0]])
  %tl = "tf.EmptyTensorList"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %while:2 = "tf.WhileRegion"(%tl, %tl) ({
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>, %barg2: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
      %cond = "tf.false"():()-> tensor<i1>
      "tf.Yield"(%cond) : (tensor<i1>) -> ()
  }, {
    ^bb0(%barg1: tensor<!tf_type.variant<tensor<?x1xf32>>>, %barg2: tensor<!tf_type.variant<tensor<?x1xf32>>>): // no predeceessors
    "tf.Yield"(%barg1, %barg2) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>) -> ()
  }) {is_stateless = false} : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>) -> (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.variant<tensor<?x1xf32>>>)
  func.return
}

// CHECK: no_while_region_op
func.func @no_while_region_op() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: TensorListReserve
  // CHECK: TensorListReserve
  %tl = "tf.TensorListReserve"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_1 = "tf._SomeOtherOp"() : () -> tensor<8x1xf32>
  %tl_set_item = "tf.TensorListSetItem"(%tl, %one, %elem_1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<8x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return
}

// CHECK: no_while_region_op_empty_tensor_list
func.func @no_while_region_op_empty_tensor_list() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: EmptyTensorList
  // CHECK: EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_1 = "tf._SomeOtherOp"() : () -> tensor<8x1xf32>
  %tl_set_item = "tf.TensorListSetItem"(%tl, %one, %elem_1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<8x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return
}

// CHECK: use_two_sep_ops
func.func @use_two_sep_ops() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: TensorListReserve
  // CHECK: TensorListReserve
  // CHECK: TensorListReserve
  %tl = "tf.TensorListReserve"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_1 = "tf._FirstOp"() : () -> tensor<8x1xf32>
  %tl_set_item = "tf.TensorListSetItem"(%tl, %one, %elem_1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<8x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_2 = "tf._SecondOp"() : () -> tensor<16x1xf32>
  %tl_set_item2 = "tf.TensorListSetItem"(%tl, %one, %elem_2) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<16x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return
}

// CHECK: use_two_sep_ops_empty_tensor_list
func.func @use_two_sep_ops_empty_tensor_list() {
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: EmptyTensorList
  // CHECK: EmptyTensorList
  // CHECK: EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_1 = "tf._FirstOp"() : () -> tensor<8x1xf32>
  %tl_set_item = "tf.TensorListSetItem"(%tl, %one, %elem_1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<8x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  %elem_2 = "tf._SecondOp"() : () -> tensor<16x1xf32>
  %tl_set_item2 = "tf.TensorListSetItem"(%tl, %one, %elem_2) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<16x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return
}
