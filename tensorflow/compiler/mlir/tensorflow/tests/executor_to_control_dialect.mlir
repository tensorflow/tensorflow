// RUN: tf-opt -tf-executor-to-control-conversion %s | FileCheck %s --dump-input=fail
// CHECK-LABEL: func @LoopTest() {
func @LoopTest() {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      %cst = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %1:2 = tf_executor.Enter %0#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T = "tfdtype$DT_INT32", device = "", name = "while/Enter"}
    %2 = tf_executor.island {
      "tf.NoOp"() {device = "", name = "cluster/pivot"} : () -> ()
      tf_executor.yield
    }
    %3:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"}
    %4:3 = tf_executor.Merge %3#0, %1#0 : tensor<*xi32> {N = 2 : i64, T = "tfdtype$DT_INT32", device = "", name = "while/Merge"}
    %5:2 = tf_executor.island(%4#2) {
      %cst = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "while/Less/y", value = dense<2> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %14 = "tf.Less"(%4#0, %5#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Less"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      tf_executor.yield %14 : tensor<*xi1>
    }
    %7:2 = tf_executor.LoopCond %6#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {device = "", name = "while/LoopCond"}
    %8:3 = tf_executor.Switch %4#0, %7#0 : tensor<*xi32> {T = "tfdtype$DT_INT32", _class = ["loc = @while/Merge"], device = "", name = "while/Switch"}
    %9:2 = tf_executor.Exit %8#0 : tensor<*xi32> {T = "tfdtype$DT_INT32", device = "", name = "while/Exit"}
    %10:2 = tf_executor.island {
      %14 = "tf.Identity"(%8#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Identity"} : (tensor<*xi32>) -> tensor<*xi32>
      tf_executor.yield %14 : tensor<*xi32>
    }
    %11:2 = tf_executor.island(%10#1) {
      %cst = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "while/Add/y", value = dense<3> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %12:2 = tf_executor.island {
      %14 = "tf.Add"(%10#0, %11#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Add"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %14 : tensor<*xi32>
    }
    %13 = tf_executor.ControlTrigger %2, %12#1, %9#1 {_tpu_replicate = "cluster", device = "", name = "gradients/while/mul_2_Da30D05wlPU_grad/SymbolicGradient/b_sync"}
    tf_executor.NextIteration.Sink [%3#1] %12#0, %13 : tensor<*xi32> {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"}
    tf_executor.fetch
  }
  return
}
// CHECK-NEXT:   %[[CONST:[0-9]*]]:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> (tensor<i32>, !_tf.control)
// CHECK-NEXT:   %[[ENTER:[0-9]*]]:2 = "_tf.Enter"(%[[CONST]]#0) {T = "tfdtype$DT_INT32", device = "", frame_name = "while/while_context", is_constant = false, name = "while/Enter", parallel_iterations = 10 : i64} : (tensor<i32>) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[NOOP:[0-9]*]] = "_tf.NoOp"() {device = "", name = "cluster/pivot"} : () -> !_tf.control
// CHECK-NEXT:   %[[SOURCE:[0-9]*]]:2 = "_tf.NextIteration.source"() {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"} : () -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[MERGE:[0-9]*]]:3 = "_tf.Merge"(%[[SOURCE]]#0, %[[ENTER]]#0) {N = 2 : i64, T = "tfdtype$DT_INT32", device = "", name = "while/Merge"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<i32>, !_tf.control)
// CHECK-NEXT:   %[[CONST_LESS:[0-9]*]]:2 = "_tf.Const"(%[[MERGE]]#2) {device = "", dtype = "tfdtype$DT_INT32", name = "while/Less/y", value = dense<2> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
// CHECK-NEXT:   %[[LESS:[0-9]*]]:2 = "_tf.Less"(%[[MERGE]]#0, %[[CONST_LESS]]#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Less"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>, !_tf.control)
// CHECK-NEXT:   %[[COND:[0-9]*]]:2 = "_tf.LoopCond"(%[[LESS]]#0) {device = "", name = "while/LoopCond"} : (tensor<*xi1>) -> (tensor<i1>, !_tf.control)
// CHECK-NEXT:   %[[SWITCH:[0-9]*]]:3 = "_tf.Switch"(%[[MERGE]]#0, %[[COND]]#0) {T = "tfdtype$DT_INT32", _class = ["loc = @while/Merge"], device = "", name = "while/Switch"} : (tensor<*xi32>, tensor<i1>) -> (tensor<*xi32>, tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[EXIT:[0-9]*]]:2 = "_tf.Exit"(%[[SWITCH]]#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Exit"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[IDENTITY:[0-9]*]]:2 = "_tf.Identity"(%[[SWITCH]]#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Identity"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[CONST_ADD:[0-9]*]]:2 = "_tf.Const"(%[[IDENTITY]]#1) {device = "", dtype = "tfdtype$DT_INT32", name = "while/Add/y", value = dense<3> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control)
// CHECK-NEXT:   %[[ADD:[0-9]*]]:2 = "_tf.Add"(%[[IDENTITY]]#0, %[[CONST_ADD]]#0) {T = "tfdtype$DT_INT32", device = "", name = "while/Add"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT:   %[[CT:[0-9]*]] = "_tf.ControlTrigger"(%[[NOOP]], %[[ADD]]#1, %[[EXIT]]#1) {_tpu_replicate = "cluster", device = "", name = "gradients/while/mul_2_Da30D05wlPU_grad/SymbolicGradient/b_sync"} : (!_tf.control, !_tf.control, !_tf.control) -> !_tf.control
// CHECK-NEXT:   %[[SINK:[0-9]*]] = "_tf.NextIteration.sink"(%[[ADD]]#0, %[[CT]]) {T = "tfdtype$DT_INT32", device = "", id = 0 : i64, name = "while/NextIteration"} : (tensor<*xi32>, !_tf.control) -> !_tf.control
// CHECK-NEXT:   return

// -----

// CHECK-LABEL: func @multiple_ops_region
func @multiple_ops_region(%arg0 : tensor<*xi32>, %arg1 : tensor<i32>) {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      // The 4 operations are independent, but the current conversion will add
      // control dependencies conservatively.
      %1 = "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add1"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add2"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %3 = "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add3"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %4 = "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add4"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %4 : tensor<*xi32>
    }
    tf_executor.fetch
  }
  return
}
// CHECK-NEXT: %[[ADD1:[0-9]*]]:2 = "_tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add1"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT: %[[ADD2:[0-9]*]]:2 = "_tf.Add"(%arg0, %arg1, %[[ADD1]]#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add2"} : (tensor<*xi32>, tensor<i32>, !_tf.control) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT: %[[ADD3:[0-9]*]]:2 = "_tf.Add"(%arg0, %arg1, %[[ADD2]]#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add3"} : (tensor<*xi32>, tensor<i32>, !_tf.control) -> (tensor<*xi32>, !_tf.control)
// CHECK-NEXT: %[[ADD4:[0-9]*]]:2 = "_tf.Add"(%arg0, %arg1, %[[ADD3]]#1) {T = "tfdtype$DT_INT32", device = "", name = "while/Add4"} : (tensor<*xi32>, tensor<i32>, !_tf.control) -> (tensor<*xi32>, !_tf.control)

// -----

// CHECK-LABEL: func @switchN(
func @switchN(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {
    // CHECK: [[S1:%.*]]:6 = "_tf._SwitchN"(%arg1, %arg0) {num_outs = 5 : i64}
    %1:6 = tf_executor.SwitchN %arg1, %arg0 of 5 : tensor<*xf32>
    // CHECK: "_tf._SwitchN"(%arg1, %arg0, [[S1]]#5) {num_outs = 12 : i64}
    %2:13 = tf_executor.SwitchN %arg1, %arg0 of 12 (%1#5) : tensor<*xf32>
    tf_executor.fetch %2#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// -----

// Test if tf_executor dialect ops with Ref types are mapped correctly to the ops in control dialect.
// CHECK-LABEL: func @ref_tf_executor_ops
func @ref_tf_executor_ops(%arg0: tensor<4x!tf.f32ref>, %arg1: tensor<4x!tf.f32ref>, %arg3: tensor<i32>, %arg4: tensor<i1> ) -> tensor<4x!tf.f32ref> {
  %result = tf_executor.graph {
          // CHECK: _tf.Enter
          %0:2 = tf_executor.Enter %arg0 frame "while/while_context" : (tensor<4x!tf.f32ref>) -> (tensor<4x!tf.f32ref>, !tf_executor.control)
          // CHECK: _tf.Exit
          %1:2 = tf_executor.Exit %arg0 : tensor<4x!tf.f32ref>
          // CHECK: _tf.Switch
          %2:3 = tf_executor.Switch %arg0, %arg4 : (tensor<4x!tf.f32ref>, tensor<i1>) -> (tensor<4x!tf.f32ref>, tensor<4x!tf.f32ref>, !tf_executor.control)
          // CHECK: _tf.Merge
          %3:3 = tf_executor.Merge %arg0, %arg1 : (tensor<4x!tf.f32ref>, tensor<4x!tf.f32ref>) -> (tensor<4x!tf.f32ref>, tensor<i32>, !tf_executor.control)
          // CHECK: _tf.NextIteration.source
          %4:3 = tf_executor.NextIteration.Source : tensor<4x!tf.f32ref>
          // CHECK: _tf.NextIteration.sink
          tf_executor.NextIteration.Sink [%4#1] %4#0 : tensor<4x!tf.f32ref>
          tf_executor.fetch %0#0 : tensor<4x!tf.f32ref>
  }
  return %result : tensor<4x!tf.f32ref>
}

// -----

// Tests if empty island with just one control dependency input and output is
// handled correctly.
// CHECK-LABEL: func @empty_island_control_dep_only
func @empty_island_control_dep_only() -> tensor<i32> {
  %fetch = tf_executor.graph {
    %0:2 = tf_executor.island {
      %4 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %4 : tensor<i32>
    }
    // CHECK-NEXT: %[[CONST1:[0-9]*]]:2 = "_tf.Const"()
    // CHECK-SAME: () -> (tensor<i32>, !_tf.control)
    %1:2 = tf_executor.island {
      %5 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %5 : tensor<i32>
    }
    // CHECK-NEXT: %[[CONST2:[0-9]*]]:2 = "_tf.Const"()
    // CHECK-SAME: () -> (tensor<i32>, !_tf.control)
    %2 = tf_executor.island(%0#1) {
      tf_executor.yield
    }
    %3:2 = tf_executor.island(%2, %1#1) {
      %6 = "tf.Add"(%0#0, %1#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    // CHECK-NEXT: %[[ADD:[0-9]*]]:2 = "_tf.Add"(%[[CONST1]]#0, %[[CONST2]]#0, %[[CONST1]]#1, %[[CONST2]]#1)
    // CHECK-SAME: (tensor<i32>, tensor<i32>, !_tf.control, !_tf.control) -> (tensor<i32>, !_tf.control)
    tf_executor.fetch %3#0 : tensor<i32>
  }
  return %fetch : tensor<i32>
}

// -----

// Tests if empty island with multiple control inputs will be replaced with a
// no-op.
// CHECK-LABEL: func @empty_island_multi_control_inputs
func @empty_island_multi_control_inputs() -> tensor<i32> {
  %fetch = tf_executor.graph {
    %0:2 = tf_executor.island {
      %4 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %4 : tensor<i32>
    }
    // CHECK-NEXT: %[[CONST1:[0-9]*]]:2 = "_tf.Const"()
    // CHECK-SAME: () -> (tensor<i32>, !_tf.control)
    %1:2 = tf_executor.island {
      %5 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %5 : tensor<i32>
    }
    // CHECK-NEXT: %[[CONST2:[0-9]*]]:2 = "_tf.Const"()
    // CHECK-SAME: () -> (tensor<i32>, !_tf.control)
    %2 = tf_executor.island(%0#1, %1#1) {
      tf_executor.yield
    }
    // CHECK-NEXT: %[[NOOP:[0-9]*]] = "_tf.NoOp"(%[[CONST1]]#1, %[[CONST2]]#1)
    // CHECK-SAME: (!_tf.control, !_tf.control) -> !_tf.control
    %3:2 = tf_executor.island(%2) {
      %6 = "tf.Add"(%0#0, %1#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    // CHECK-NEXT: %[[ADD:[0-9]*]]:2 = "_tf.Add"(%[[CONST1]]#0, %[[CONST2]]#0, %[[NOOP]])
    // CHECK-SAME: (tensor<i32>, tensor<i32>, !_tf.control) -> (tensor<i32>, !_tf.control)
    tf_executor.fetch %3#0 : tensor<i32>
  }
  return %fetch : tensor<i32>
}
