// RUN: tf-opt %s -split-input-file -tf-executor-graph-pruning | FileCheck %s

// Two islands chained by data-flow contributing to the graph return are
// preserved.
// CHECK-LABEL: func @chained_islands(
func @chained_islands(%arg0 : i32) -> i32 {
// CHECK: island
// CHECK: island
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : i32
    }
    %2:2 = tf_executor.island {
      tf_executor.yield %1#0 : i32
    }
    tf_executor.fetch %2#0 : i32
  }
  return %0 : i32
}

// Check that an unused island that doesn't contribute to the fetch is removed.
// CHECK-LABEL: func @dead_island(
func @dead_island(%arg0 : i32) -> i32 {
// CHECK: tf_executor.island
// CHECK-NOT: tf_executor.island
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %a = "tf.opA"(%arg0) : (i32) -> i32
      %b = "tf.opB"(%a) : (i32) -> i32
      tf_executor.yield %b : i32
    }
    %2:2 = tf_executor.island {
      %a = "tf.opA"(%1#0) : (i32) -> i32
      tf_executor.yield %a : i32
    }
    tf_executor.fetch %1#0 : i32
  }
  return %0 : i32
}


// Check that NextIteration.sink node isn't deleted when the source is still
// used, even though it does not have any result.
// CHECK-LABEL: func @nextiteration_sink_preserved(
func @nextiteration_sink_preserved(%arg0 : i32) -> i32 {
// CHECK: tf_executor.NextIteration.Source
// CHECK: tf_executor.NextIteration.Sink
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : i32
    tf_executor.NextIteration.Sink[%1#1] %1#0 : i32
    tf_executor.fetch %1#0 : i32
  }
  return %0 : i32
}

// Check that NextIteration.sink node is deleted when the source does not have
// any user other than the sink.
// CHECK-LABEL: func @nextiteration_deleted(
func @nextiteration_deleted(%arg0 : i32) -> i32 {
// CHECK-NOT: tf_executor.NextIteration.Source
// CHECK-NOT: tf_executor.NextIteration.Sink
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : i32
    // intentionally take an output dependency on the source here.
    tf_executor.NextIteration.Sink[%1#1] %1#0 : i32
    tf_executor.fetch %arg0 : i32
  }
  return %0 : i32
}

// Check that NextIteration.source/sink ops and associated ops are deleted when
// associated loop is unreachable.
// CHECK-LABEL: func @unreachable_loop
func @unreachable_loop(%arg0 : i32) {
// CHECK:      tf_executor.graph
// CHECK-NEXT:   tf_executor.fetch
  tf_executor.graph {
    %0:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.Enter %1#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T = "tfdtype$DT_INT32"}
    %3:3 = tf_executor.Merge %2#0, %0#0 : tensor<*xi32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    %4:2 = tf_executor.island(%3#2) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %5:2 = tf_executor.island wraps "tf.Less"(%3#0, %4#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
    %6:2 = tf_executor.LoopCond %5#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {}
    %7:3 = tf_executor.Switch %3#0, %6#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %8:2 = tf_executor.Exit %7#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %9:2 = tf_executor.island wraps "tf.Identity"(%7#1) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> tensor<*xi32>
    %10:2 = tf_executor.island(%9#1) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %11:2 = tf_executor.island wraps "tf.Add"(%9#0, %10#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    tf_executor.NextIteration.Sink [%0#1] %11#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    tf_executor.fetch %arg0 : i32
  }
  return
}

// Check that NextIteration.sink and associated ops are not deleted when
// associated loop is reachable.
// CHECK-LABEL: func @reachable_loop
func @reachable_loop() {
// CHECK: tf_executor.NextIteration.Source
// CHECK: "tf.Const"
// CHECK: tf_executor.Enter
// CHECK: tf_executor.Merge
// CHECK: "tf.Const"
// CHECK: "tf.Less"
// CHECK: tf_executor.LoopCond
// CHECK: tf_executor.Switch
// CHECK: %[[EXIT:.*]], %{{.*}} = tf_executor.Exit
// CHECK: "tf.Identity"
// CHECK: "tf.Const"
// CHECK: "tf.Add"
// CHECK: tf_executor.NextIteration.Sink
// CHECK: tf_executor.fetch %[[EXIT]]
  %0 = tf_executor.graph {
    %0:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.Enter %1#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T = "tfdtype$DT_INT32"}
    %3:3 = tf_executor.Merge %2#0, %0#0 : tensor<*xi32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    %4:2 = tf_executor.island(%3#2) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %5:2 = tf_executor.island wraps "tf.Less"(%3#0, %4#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
    %6:2 = tf_executor.LoopCond %5#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {}
    %7:3 = tf_executor.Switch %3#0, %6#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %8:2 = tf_executor.Exit %7#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    %9:2 = tf_executor.island wraps "tf.Identity"(%7#1) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> tensor<*xi32>
    %10:2 = tf_executor.island(%9#1) wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %11:2 = tf_executor.island wraps "tf.Add"(%9#0, %10#0) {T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    tf_executor.NextIteration.Sink [%0#1] %11#0 : tensor<*xi32> {T = "tfdtype$DT_INT32"}
    tf_executor.fetch %8#0 : tensor<*xi32>
  }
  return
}

// Check that ops leading to a fetch via a control are not removed.
// CHECK-LABEL: func @control_fetch
func @control_fetch(%arg0 : i32) {
// CHECK: tf_executor.island
// CHECK: tf_executor.island
// CHECK: tf_executor.island
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : i32
    }
    %2 = tf_executor.island(%0) {
      %a = "tf.opA"(%1#0) : (i32) -> i32
      tf_executor.yield
    }
    tf_executor.fetch %2 : !tf_executor.control
  }
  return
}

// -----

// Check that a function that is named "main" and does not have the
// "tf.entry_function" attribute defined is ignored by the pruning pass: this
// could be a V1 graph imported without feed/fetch/target nodes.
// CHECK-LABEL: func @main(
func @main() {
// CHECK: tf_executor.island
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// -----

// Check that a function that is named "main" and does have the
// "tf.entry_function" attribute defined with no feed/fetch/target nodes is
// pruned.
// CHECK-LABEL: func @main(
func @main() attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = ""}} {
// CHECK-NOT: tf_executor.island
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// -----

// Check that an op with must-execute effect is not pruned, even if it is
// unreachable.
func @must_execute_op() -> () {
// CHECK: tf_executor.graph
// CHECK: tf_executor.island
// CHECK: tf._InternalTestMustExecuteTrait_
  tf_executor.graph {
    %1 = tf_executor.island {
      "tf._InternalTestMustExecuteTrait_"() : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

