// RUN: tf-opt %s -tf-raise-control-flow -split-input-file | FileCheck %s

// Test that we remove underscores.

// CHECK-LABEL: func @testSimpleAddsAndIdentity(%arg0: tensor<*xf32>)
func @testSimpleAddsAndIdentity(tensor<*xf32>) -> tensor<*xf32> {
^bb0(%0: tensor<*xf32>):

  // CHECK: %0 = "tf.Identity"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "_tf.Identity"(%0) : (tensor<*xf32>) -> tensor<*xf32>

  // CHECK: %1 = "tf.Add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "_tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: %2 = "tf.Add"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "_tf.Add"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %2 : tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: func @testAddWithControlDependency(%arg0: tensor<*xf32>)
func @testAddWithControlDependency(tensor<*xf32>) -> tensor<*xf32> {
^bb0(%0: tensor<*xf32>):

  // CHECK: %0 = "tf.Identity"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %1:2 = "_tf.Identity"(%0) : (tensor<*xf32>) -> (tensor<*xf32>, !_tf.control)

  // CHECK: %1 = "tf.Add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2:2 = "_tf.Add"(%0, %0, %1#1) : (tensor<*xf32>, tensor<*xf32>, !_tf.control) -> (tensor<*xf32>, !_tf.control)

  // CHECK: %2 = "tf.Add"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3:2 = "_tf.Add"(%1#0, %2, %1#1, %2#1) : (tensor<*xf32>, tensor<*xf32>, !_tf.control, !_tf.control) -> (tensor<*xf32>, !_tf.control)

  // CHECK: return %2 : tensor<*xf32>
  return %3 : tensor<*xf32>
}

// TODO(clattner): simplify and expand these tests.  This is mostly a placeholder.
func @LoopTest() {
  %0:2 = "_tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> (tensor<i32>, !_tf.control)
  %1:2 = "_tf.Enter"(%0#0) {device = "", name = "while/Enter", T = "tfdtype$DT_INT32", frame_name = "while/while_context", is_constant = false, parallel_iterations = 10} : (tensor<i32>) -> (tensor<*xi32>, !_tf.control)

  %11:2 = "_tf.NextIteration.source"() {device = "", name = "while/NextIteration", T = "tfdtype$DT_INT32", id = 0} : () -> (tensor<*xi32>, !_tf.control)

  %2:3 = "_tf.Merge"(%11#0, %1#0) {device = "", name = "while/Merge", N = 2, T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<i32>, !_tf.control)
  %3:2 = "_tf.Const"(%2#2) {device = "", name = "while/Less/y", dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
  %4:2 = "_tf.Less"(%2#0, %3#0) {device = "", name = "while/Less", T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>, !_tf.control)
  %5:2 = "_tf.LoopCond"(%4#0) {device = "", name = "while/LoopCond"} : (tensor<*xi1>) -> (tensor<i1>, !_tf.control)
  %6:3 = "_tf.Switch"(%2#0, %5#0) {device = "", name = "while/Switch", T = "tfdtype$DT_INT32", _class = ["loc:@while/Merge"]} : (tensor<*xi32>, tensor<i1>) -> (tensor<*xi32>, tensor<*xi32>, !_tf.control)
  %7:2 = "_tf.Exit"(%6#0) {device = "", name = "while/Exit", T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
  %8:2 = "_tf.Identity"(%6#1) {device = "", name = "while/Identity", T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
  %9:2 = "_tf.Const"(%8#1) {device = "", name = "while/Add/y", dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
  %10:2 = "_tf.Add"(%8#0, %9#0) {device = "", name = "while/Add", T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, !_tf.control)
  %ctl = "_tf.NextIteration.sink"(%10#0) {device = "", name = "while/NextIteration", T = "tfdtype$DT_INT32", id = 0} : (tensor<*xi32>) -> (!_tf.control)
  return
}
