// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tensor-list-ops-decomposition | FileCheck %s

// Test push and pop on a tensor list which is initially empty.

// CHECK-LABEL: func @main
func.func @main() -> (tensor<f32>, tensor<i32>) {
  // CHECK-NEXT: "tf.Const"() <{value = dense<> : tensor<0xi32>}>
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NEXT: "tf.Const"() <{value = dense<10> : tensor<i32>}>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[ZERO_SCALAR:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: %[[CAST_ZERO:.*]] = "tf.Cast"(%[[ZERO_SCALAR]]) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: %[[CONST10:.*]] = "tf.Const"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[CAST_ZERO]], %[[CONST10]]) : (tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[ZERO:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %id = "tf.Identity"(%tl) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[PUSHVAL:.*]] = "tf._SomeOp"()
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NEXT: %[[UPDATE_SHAPE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[UPDATE_SLICE:.*]] = "tf.Reshape"(%[[PUSHVAL]], %[[UPDATE_SHAPE]]) : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"(%[[BROADCAST]], %[[UPDATE_SLICE]], %[[ZERO]]) : (tensor<10xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[NEW_SIZE:.*]] = "tf.AddV2"(%[[ZERO]], %[[CONST1]]) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %push = "tf.TensorListPushBack"(%id, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[COPY:.*]] = "tf.Identity"(%[[UPDATE]])
  // CHECK-NEXT: %[[CONST1_1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SUB:.*]] = "tf.Sub"(%[[NEW_SIZE]], %[[CONST1_1]])
  // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NEXT: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  %pop:2 = "tf.TensorListPopBack"(%push, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NEXT: %[[SCALAR_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}>
  // CHECK-NEXT: %[[LENGTH:.*]] = "tf.Reshape"(%[[NEW_SIZE]], %[[SCALAR_SHAPE]])
  %length = "tf.TensorListLength"(%push) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<i32>
  // CHECK-NEXT: return %[[ELEM]], %[[LENGTH]] : tensor<f32>, tensor<i32>
  func.return %pop#1, %length: tensor<f32>, tensor<i32>
}

// -----

// Test get and set, and other operations on a tensor list which has reserved
// initial size.

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i32>) -> (tensor<f32>, tensor<10xf32>, tensor<i32>)
func.func @main(%arg0: tensor<i32>) -> (tensor<f32>, tensor<10xf32>, tensor<i32>) {
  // CHECK-NEXT: "tf.Const"() <{value = dense<> : tensor<0xi32>}>
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NEXT: %[[NUM:.*]] = "tf.Const"() <{value = dense<10> : tensor<i32>}>
  %num = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[ZERO_SCALAR:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: %[[CAST_ZERO:.*]] = "tf.Cast"(%[[ZERO_SCALAR]]) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: %[[CONST10:.*]] = "tf.Const"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[CAST_ZERO]], %[[CONST10]]) : (tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[SIZE_SHAPE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
  // CHECK-NEXT: %[[SIZE:.*]] = "tf.Reshape"(%[[NUM]], %[[SIZE_SHAPE]])
  %tl = "tf.TensorListReserve"(%elem_shape, %num) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[SETVAL:.*]] = "tf._SomeOp"()
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NEXT: %[[SIZE_SHAPE1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
  // CHECK-NEXT: %[[SET_INDEX:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_SHAPE1]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[UPDATE_SHAPE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[UPDATE_SLICE:.*]] = "tf.Reshape"(%[[SETVAL]], %[[UPDATE_SHAPE]]) : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"(%[[BROADCAST]], %[[UPDATE_SLICE]], %[[SET_INDEX]]) : (tensor<10xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<10xf32>
  %set = "tf.TensorListSetItem"(%tl, %arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[SIZE_SHAPE2:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
  // CHECK-NEXT: %[[GET_INDEX:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_SHAPE2]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = "tf.Slice"(%[[UPDATE]], %[[GET_INDEX]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NEXT: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  %get = "tf.TensorListGetItem"(%set, %arg0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NEXT: %[[ADDN:.*]] = "tf.AddN"(%[[UPDATE]], %[[BROADCAST]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %addn = "tf.AddN"(%set, %tl) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[ZEROS_LIKE:.*]] = "tf.ZerosLike"(%[[ADDN]]) : (tensor<10xf32>) -> tensor<10xf32>
  %zeros-like = "tf.ZerosLike"(%addn) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[ADDN2:.*]] = "tf.AddN"(%[[ADDN]], %[[ZEROS_LIKE]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %addn2 = "tf.AddN"(%addn, %zeros-like) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  %stack = "tf.TensorListStack"(%addn2, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[LEN:.*]] = "tf.Const"() <{value = dense<10> : tensor<i32>}> : () -> tensor<i32>
  %length = "tf.TensorListLength"(%addn2) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<i32>
  // CHECK-NEXT: return %[[ELEM]], %[[ADDN2]], %[[LEN]] : tensor<f32>, tensor<10xf32>, tensor<i32>
  func.return %get, %stack, %length : tensor<f32>, tensor<10xf32>, tensor<i32>
}

// -----

// Test get on a tensor list created from a tensor.

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<10xf32>) -> tensor<f32>
func.func @main(%arg0: tensor<i32>, %arg1: tensor<10xf32>) -> tensor<f32> {
  // CHECK-NEXT: "tf.Const"() <{value = dense<> : tensor<0xi32>}>
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NEXT: %[[BUFFER:.*]] = "tf.Identity"(%[[ARG1]]) : (tensor<10xf32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[SIZE:.*]] = "tf.Const"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
  %tl = "tf.TensorListFromTensor"(%arg1, %elem_shape) : (tensor<10xf32>, tensor<0xi32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK-NEXT: %[[SIZE_SHAPE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
  // CHECK-NEXT: %[[GET_INDEX:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_SHAPE]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = "tf.Slice"(%[[BUFFER]], %[[GET_INDEX]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NEXT: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  %get = "tf.TensorListGetItem"(%tl, %arg0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NEXT: return %[[ELEM]] : tensor<f32>
  func.return %get: tensor<f32>
}

// -----

// Test tensor list element shape op.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<10x8x9xf32>) -> tensor<2xi64> {
  %elem_shape = "tf.Const"() {value = dense<[8, 9]> : tensor<2xi32>} : () -> tensor<2xi32>
  %tl = "tf.TensorListFromTensor"(%arg0, %elem_shape) : (tensor<10x8x9xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<8x9xf32>>>
  // CHECK: %[[SHAPE:.*]] = "tf.Const"() <{value = dense<[8, 9]> : tensor<2xi64>}> : () -> tensor<2xi64>
  %shape = "tf.TensorListElementShape"(%tl) : (tensor<!tf_type.variant<tensor<8x9xf32>>>) -> tensor<2xi64>
  // CHECK-NEXT: return %[[SHAPE]] : tensor<2xi64>
  func.return %shape: tensor<2xi64>
}

// -----

// Test tensor list gather op.

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x8x9xf32>, %[[ARG1:.*]]: tensor<3xi32>) -> tensor<3x8x9xf32>
func.func @main(%arg0: tensor<10x8x9xf32>, %arg1: tensor<3xi32>) -> tensor<3x8x9xf32> {
  %elem_shape = "tf.Const"() {value = dense<[8, 9]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[BUFFER:.*]] = "tf.Identity"(%[[ARG0]]) : (tensor<10x8x9xf32>) -> tensor<10x8x9xf32>
  %tl = "tf.TensorListFromTensor"(%arg0, %elem_shape) : (tensor<10x8x9xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<8x9xf32>>>
  // CHECK: %[[AXIS:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  // CHECK: %[[GATHER:.*]] = "tf.GatherV2"(%[[BUFFER]], %[[ARG1]], %[[AXIS]]) : (tensor<10x8x9xf32>, tensor<3xi32>, tensor<i32>) -> tensor<3x8x9xf32>
  %gather = "tf.TensorListGather"(%tl, %arg1, %elem_shape) : (tensor<!tf_type.variant<tensor<8x9xf32>>>, tensor<3xi32>, tensor<2xi32>) -> tensor<3x8x9xf32>
  // CHECK-NEXT: return %[[GATHER]] : tensor<3x8x9xf32>
  func.return %gather: tensor<3x8x9xf32>
}

// -----

// Test scatter into existing tensor list.

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10x8x9xf32>, %[[ARG1:.*]]: tensor<5xi32>, %[[ARG2:.*]]: tensor<5x8x9xf32>) -> tensor<10x8x9xf32>
func.func @main(%arg0: tensor<10x8x9xf32>, %arg1: tensor<5xi32>, %arg2: tensor<5x8x9xf32>) -> tensor<10x8x9xf32> {
  %elem_shape = "tf.Const"() {value = dense<[8, 9]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[BUFFER:.*]] = "tf.Identity"(%[[ARG0]]) : (tensor<10x8x9xf32>) -> tensor<10x8x9xf32>
  %tl = "tf.TensorListFromTensor"(%arg0, %elem_shape) : (tensor<10x8x9xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<8x9xf32>>>
  // CHECK: %[[IND_SHAPE:.*]] = "tf.Const"() <{value = dense<[5, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK: %[[IND_RESHPE:.*]] = "tf.Reshape"(%[[ARG1]], %[[IND_SHAPE]]) : (tensor<5xi32>, tensor<2xi32>) -> tensor<5x1xi32>
  // CHECK: %[[SC:.*]] = "tf.TensorScatterUpdate"(%[[BUFFER]], %[[IND_RESHPE]], %[[ARG2]]) <{bad_indices_policy = ""}> : (tensor<10x8x9xf32>, tensor<5x1xi32>, tensor<5x8x9xf32>) -> tensor<10x8x9xf32>
  %scatter = "tf.TensorListScatterIntoExistingList"(%tl, %arg2, %arg1) : (tensor<!tf_type.variant<tensor<8x9xf32>>>, tensor<5x8x9xf32>, tensor<5xi32>) -> tensor<!tf_type.variant<tensor<8x9xf32>>>
  %stack = "tf.TensorListStack"(%scatter, %elem_shape) : (tensor<!tf_type.variant<tensor<8x9xf32>>>, tensor<2xi32>) -> tensor<10x8x9xf32>
  // CHECK: return %[[SC]] : tensor<10x8x9xf32>
  func.return %stack : tensor<10x8x9xf32>
}

// -----

// Tests while loop.

// CHECK-LABEL: func @main
func.func @main() -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %1:2 = "tf.While"(%tl, %max_size) {
    body = @while_body, cond = @while_cond, device = "", is_stateless = false}
       : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>)
  // CHECK: "tf.Slice"
  %pop:2 = "tf.TensorListPopBack"(%1#0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NOT: tf.TensorListPopBack
  // CHECK: return
  func.return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<10xf32>, %[[BARG1:.*]]: tensor<i32>, %[[BARG2:.*]]: tensor<1xi32>)
func.func @while_body(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>) {
  // CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[BARG1]], %[[CONST1]])
  %sub = "tf.Sub"(%arg1, %const1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPushBack"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[BARG2]], %[[CONST1]])
  // CHECK-NOT: "tf.TensorListPushBack"
  %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: return %[[UPDATE]], %[[SUB]], %[[ADD]]
  func.return %push, %sub : tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>
}
// CHECK: func @while_cond(%[[CARG0:.*]]: tensor<10xf32>, %[[CARG1:.*]]: tensor<i32>, %[[CARG2:.*]]: tensor<1xi32>)
func.func @while_cond(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return %[[CARG1]]
  func.return %arg1 : tensor<i32>
}

// -----

// Tests IfOp.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i1>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %if_op = "tf.If"(%arg0, %tl) {then_branch = @if_then, else_branch = @if_else, is_stateless = false}
    : (tensor<i1>, tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.Slice"
  %pop:2 = "tf.TensorListPopBack"(%if_op, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NOT: tf.TensorListPopBack
  // CHECK: return
  func.return
}
// CHECK: func @if_then(%[[TARG0:.*]]: tensor<10xf32>, %[[TARG1:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func @if_then(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPushBack"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[TARG1]], %[[CONST1]])
  // CHECK-NOT: "tf.TensorListPushBack"
  %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: return %[[UPDATE]], %[[ADD]]
  func.return %push : tensor<!tf_type.variant<tensor<f32>>>
}
// CHECK: func @if_else(%[[EARG0:.*]]: tensor<10xf32>, %[[EARG1:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func @if_else(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NOT: "tf.TensorListPopBack"
  // CHECK: %[[COPY:.*]] = "tf.Identity"(%[[EARG0]])
  // CHECK: %[[CONST1_1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[EARG1]], %[[CONST1_1]])
  // CHECK: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPopBack"
  %pop:2 = "tf.TensorListPopBack"(%arg0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK: return %[[COPY]], %[[SUB]]
  func.return %pop#0 : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// Tests CaseOp.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %case_op = "tf.Case"(%arg0, %tl) {branches = [@branch_0, @branch_1, @branch_2], is_stateless = false}
    : (tensor<i32>, tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.Slice"
  %pop:2 = "tf.TensorListPopBack"(%case_op, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NOT: tf.TensorListPopBack
  // CHECK: return
  func.return
}
// CHECK: func @branch_0(%[[TARG0:.*]]: tensor<10xf32>, %[[TARG1:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func @branch_0(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPushBack"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[TARG1]], %[[CONST1]])
  // CHECK-NOT: "tf.TensorListPushBack"
  %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: return %[[UPDATE]], %[[ADD]]
  func.return %push : tensor<!tf_type.variant<tensor<f32>>>
}
// CHECK: func @branch_1(%[[EARG0:.*]]: tensor<10xf32>, %[[EARG1:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func @branch_1(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NOT: "tf.TensorListPopBack"
  // CHECK: %[[COPY:.*]] = "tf.Identity"(%[[EARG0]])
  // CHECK: %[[CONST1_1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[EARG1]], %[[CONST1_1]])
  // CHECK: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPopBack"
  %pop:2 = "tf.TensorListPopBack"(%arg0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK: return %[[COPY]], %[[SUB]]
  func.return %pop#0 : tensor<!tf_type.variant<tensor<f32>>>
}
// CHECK: func @branch_2(%[[EARG0:.*]]: tensor<10xf32>, %[[EARG1:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func @branch_2(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK-NOT: "tf.TensorListPopBack"
  // CHECK: %[[COPY:.*]] = "tf.Identity"(%[[EARG0]])
  // CHECK: %[[CONST1_1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[EARG1]], %[[CONST1_1]])
  // CHECK: %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]]) : (tensor<10xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xf32>
  // CHECK: %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK: %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]]) : (tensor<1xf32>, tensor<0xi32>) -> tensor<f32>
  // CHECK-NOT: "tf.TensorListPopBack"
  %pop:2 = "tf.TensorListPopBack"(%arg0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK: return %[[COPY]], %[[SUB]]
  func.return %pop#0 : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// CHECK-LABEL: func @main
func.func @main() -> tensor<f32> {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %while_op:2 = "tf.WhileRegion"(%tl, %size) <{is_stateless = false}> ({
  // CHECK: ^bb0(%[[CARG0:.*]]: tensor<10xf32>, %[[CARG1:.*]]: tensor<i32>, %[[CARG2:.*]]: tensor<1xi32>):
  ^bb0(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i32>):
    // CHECK:   %[[PRED:.*]] = "tf._SomeOp"()
    // CHECK:   "tf.Yield"(%[[PRED]])
    %pred = "tf._SomeOp"() : () -> tensor<i1>
    "tf.Yield"(%pred) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0(%[[CARG0:.*]]: tensor<10xf32>, %[[CARG1:.*]]: tensor<i32>, %[[CARG2:.*]]: tensor<1xi32>):
  ^bb0(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i32>):
    // CHECK:   %[[CST:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    // CHECK:   %[[SUB:.*]] = "tf.Sub"(%[[CARG1]], %[[CST]])
    // CHECK:   %[[ELEM:.*]] = "tf._SomeOp"() : () -> tensor<f32>
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %sub = "tf.Sub"(%arg1, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %elem = "tf._SomeOp"() : () -> tensor<f32>
    // CHECK-NOT: "tf.TensorListPushBack"
    // CHECK:   %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"(%[[CARG0]]
    // CHECK:   %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
    // CHECK:   %[[ADD:.*]] = "tf.AddV2"(%[[CARG2]], %[[ONE]])
    // CHECK-NOT: "tf.TensorListPushBack"
    // CHECK:   "tf.Yield"(%[[UPDATE]], %[[SUB]], %[[ADD]])
    // CHECK: })
    %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
    "tf.Yield"(%push, %sub) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>) -> ()
  }) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>)
  // CHECK: "tf.Slice"
  // CHECK-NOT: tf.TensorListPopBack
  %pop:2 = "tf.TensorListPopBack"(%while_op#0, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK: return
  func.return %pop#1 : tensor<f32>
}
// -----

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i1>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: %[[ZERO:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}>
  // CHECK: %[[ZERO_F32:.*]] = "tf.Cast"(%[[ZERO]])
  // CHECK: %[[MAX_SIZE:.*]] = "tf.Const"() <{value = dense<10> : tensor<1xi32>}>
  // CHECK: %[[BUFFER:.*]] = "tf.BroadcastTo"(%[[ZERO_F32]], %[[MAX_SIZE]])
  // CHECK: %[[BUFFER_SIZE:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}>
  // CHECK-NOT: tf.EmptyTensorList
  %if_op = "tf.IfRegion"(%arg0) ({
      %elem = "tf._SomeOp"() : () -> tensor<f32>
      // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
      // CHECK: %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
      // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[BUFFER_SIZE]], %[[ONE]])
      // CHECK-NOT: "tf.TensorListPushBack"
      %push = "tf.TensorListPushBack"(%tl, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
      "tf.Yield" (%push) : (tensor<!tf_type.variant<tensor<f32>>>) -> ()
    }, {
      // CHECK:   %[[COPY:.*]] = "tf.Identity"(%[[BUFFER]])
      // CHECK:   %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SUB:.*]] = "tf.Sub"(%[[BUFFER_SIZE]], %[[ONE]])
      // CHECK:   %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]])
      // CHECK:   %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}>
      // CHECK:   %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]])
      // CHECK-NOT: "tf.TensorListPopBack"
      %pop:2 = "tf.TensorListPopBack"(%tl, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
      // CHECK:   "tf.Yield"(%[[COPY]], %[[SUB]])
      "tf.Yield" (%pop#0) : (tensor<!tf_type.variant<tensor<f32>>>) -> ()
    })
    {is_stateless = false}
    : (tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.Slice"
  // CHECK-NOT: tf.TensorListPopBack
  %pop:2 = "tf.TensorListPopBack"(%if_op, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  func.return
}

// -----

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ZERO:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}>
  // CHECK: %[[ZERO_F32:.*]] = "tf.Cast"(%[[ZERO]])
  // CHECK: %[[MAX_SIZE:.*]] = "tf.Const"() <{value = dense<10> : tensor<1xi32>}>
  // CHECK: %[[BUFFER:.*]] = "tf.BroadcastTo"(%[[ZERO_F32]], %[[MAX_SIZE]])
  // CHECK: %[[BUFFER_SIZE:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}>
  // CHECK-NOT: tf.EmptyTensorList
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %case_op = "tf.CaseRegion"(%arg0) ({
      %elem = "tf._SomeOp"() : () -> tensor<f32>
      // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
      // CHECK: %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
      // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[BUFFER_SIZE]], %[[ONE]])
      // CHECK-NOT: "tf.TensorListPushBack"
      %push = "tf.TensorListPushBack"(%tl, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
      "tf.Yield" (%push) : (tensor<!tf_type.variant<tensor<f32>>>) -> ()
    }, {
      // CHECK:   %[[COPY:.*]] = "tf.Identity"(%[[BUFFER]])
      // CHECK:   %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SUB:.*]] = "tf.Sub"(%[[BUFFER_SIZE]], %[[ONE]])
      // CHECK:   %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]])
      // CHECK:   %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}>
      // CHECK:   %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]])
      // CHECK-NOT: "tf.TensorListPopBack"
      %pop:2 = "tf.TensorListPopBack"(%tl, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
      // CHECK:   "tf.Yield"(%[[COPY]], %[[SUB]])
      "tf.Yield" (%pop#0) : (tensor<!tf_type.variant<tensor<f32>>>) -> ()
    }, {
      // CHECK:   %[[COPY:.*]] = "tf.Identity"(%[[BUFFER]])
      // CHECK:   %[[ONE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SUB:.*]] = "tf.Sub"(%[[BUFFER_SIZE]], %[[ONE]])
      // CHECK:   %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}>
      // CHECK:   %[[SLICE:.*]] = "tf.Slice"(%[[COPY]], %[[SUB]], %[[SLICE_SIZE]])
      // CHECK:   %[[ELEM_SHAPE:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}>
      // CHECK:   %[[ELEM:.*]] = "tf.Reshape"(%[[SLICE]], %[[ELEM_SHAPE]])
      // CHECK-NOT: "tf.TensorListPopBack"
      %pop:2 = "tf.TensorListPopBack"(%tl, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
      // CHECK:   "tf.Yield"(%[[COPY]], %[[SUB]])
      "tf.Yield" (%pop#0) : (tensor<!tf_type.variant<tensor<f32>>>) -> ()
    }) {is_stateless = false}
    : (tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.Slice"
  // CHECK-NOT: tf.TensorListPopBack
  %pop:2 = "tf.TensorListPopBack"(%case_op, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  func.return
}

// -----

// Tests PartitionedCall/StatefulPartitionedCall.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i1>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  // CHECK: %[[INIT:.*]] = "tf.BroadcastTo"
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.StatefulPartitionedCall"(%[[INIT]],
  // CHECK-SAME: f = @callee_tensorlist_decomposed
  %call = "tf.StatefulPartitionedCall"(%tl, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: %[[CALL2:.*]]:2 = "tf.PartitionedCall"(%[[INIT]],
  // CHECK-SAME: f = @callee_tensorlist_decomposed
  %call2 = "tf.PartitionedCall"(%tl, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: %[[COPY:.*]] = "tf.Identity"(%[[CALL2]]#0)
  // CHECK: "tf.Slice"(%[[COPY]],
  %pop:2 = "tf.TensorListPopBack"(%call2, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NOT: tf.TensorListPopBack
  // CHECK: return
  func.return
}

// CHECK: func @callee(%[[AARG0:.*]]: tensor<!tf_type.variant<tensor<f32>>>, %[[AARG1:.*]]: tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
func.func @callee(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem = "tf._SomeOp"(%arg1) : (tensor<i1>) -> tensor<f32>
  // CHECK: "tf.TensorListPushBack"
  %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  func.return %push : tensor<!tf_type.variant<tensor<f32>>>
}

// CHECK: func private @callee_tensorlist_decomposed(%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<i1>, %[[ARG2:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
// CHECK-NOT: "tf.TensorListPushBack"
// CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
// CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ARG2]], %[[CONST1]])
// CHECK-NOT: "tf.TensorListPushBack"
// CHECK: return %[[UPDATE]], %[[ADD]]

// -----

// Tests PartitionedCall/StatefulPartitionedCall with private callee function.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i1>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  // CHECK: %[[INIT:.*]] = "tf.BroadcastTo"
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: "tf.StatefulPartitionedCall"(%[[INIT]],
  // CHECK-SAME: f = @callee
  %call = "tf.StatefulPartitionedCall"(%tl, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: %[[CALL2:.*]]:2 = "tf.PartitionedCall"(%[[INIT]],
  // CHECK-SAME: f = @callee
  %call2 = "tf.PartitionedCall"(%tl, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: %[[COPY:.*]] = "tf.Identity"(%[[CALL2]]#0)
  // CHECK: "tf.Slice"(%[[COPY]],
  %pop:2 = "tf.TensorListPopBack"(%call2, %elem_shape) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>)
  // CHECK-NOT: tf.TensorListPopBack
  // CHECK: return
  func.return
}

// CHECK: func private @callee(%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<i1>, %[[ARG2:.*]]: tensor<1xi32>) -> (tensor<10xf32>, tensor<1xi32>)
func.func private @callee(%arg0: tensor<!tf_type.variant<tensor<f32>>>, %arg1: tensor<i1>) -> tensor<!tf_type.variant<tensor<f32>>> {
  %elem = "tf._SomeOp"(%arg1) : (tensor<i1>) -> tensor<f32>

  // CHECK-NOT: "tf.TensorListPushBack"
  // CHECK: %[[UPDATE:.*]] = "tf.XlaDynamicUpdateSlice"
  // CHECK: %[[CONST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ARG2]], %[[CONST1]])
  // CHECK-NOT: "tf.TensorListPushBack"
  %push = "tf.TensorListPushBack"(%arg0, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  // CHECK: return %[[UPDATE]], %[[ADD]]
  func.return %push : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// Tests PartitionedCall op with no signature change on callee.

// CHECK-LABEL: func @main
func.func @main() {
  "tf.PartitionedCall"() {f = @callee, config = "", config_proto = "", executor_type = ""} : () -> ()
  func.return
}
// CHECK: func private @callee()
func.func @callee() {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  // CHECK: "tf.BroadcastTo"
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  func.return
}

// -----

// Tests that the pass uses the result type to infer element shape.

func.func @main(%arg0 : tensor<*xi32>)  -> () {
  // 1-D element shape with dynamic size
  %element_shape = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NOT: tf.EmptyTensorList
  // CHECK: tf.BroadcastTo
  // CHECK-SAME: tensor<10x16xf32>
  %tl0 = "tf.EmptyTensorList"(%element_shape, %max_size) : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<16xf32>>>
  // CHECK-NOT: tf.TensorListReserve
  // CHECK: tf.BroadcastTo
  // CHECK-SAME: tensor<10x32xf32>
  %tl1 = "tf.TensorListReserve"(%arg0, %max_size) : (tensor<*xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<32xf32>>>
  func.return
}

// -----

// Tests that the pass reports error on unknown maximum size.

func.func @main(%arg0: tensor<i32>) -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // expected-error @+1 {{unknown max element count}}
  %tl = "tf.EmptyTensorList"(%elem_shape, %arg0) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  func.return
}

// -----

// Tests that the pass reports error on unknown element shape.

func.func @main()  -> () {
  %elem_shape = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{unknown tensor list element shape}}
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return
}

// -----

// Tests that the pass reports error on unknown element shape.

func.func @main(%arg0: tensor<*xi32>)  -> () {
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{unknown tensor list element shape}}
  %tl = "tf.EmptyTensorList"(%arg0, %max_size) : (tensor<*xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  func.return
}

// -----

// Tests that the pass reports error on pushing elements to a fixed-size tenosr
// list.

func.func @main(%arg0: tensor<*xi32>)  -> () {
  %elem_shape = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %num = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %tl = "tf.TensorListReserve"(%elem_shape, %num) : (tensor<0xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %elem = "tf._SomeOp"() : () -> tensor<f32>
  // expected-error @+1 {{cannot push on a fixed-size tensor list}}
  %push = "tf.TensorListPushBack"(%tl, %elem) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  func.return
}
