// RUN: tf-opt -tf-executor-break-up-islands %s | FileCheck %s --dump-input=fail
// RUN: tf-opt -tf-executor-break-up-islands -tf-executor-break-up-islands %s | FileCheck %s --dump-input=fail

// All tests also test for idempotence.

func @multiple_return(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island#0, %island#1 : tensor<*xi32>, tensor<*xi32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @multiple_return
// CHECK:   %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:     tf_executor.fetch %[[ADD1]], %[[ADD2]] :
// CHECK:   }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
// CHECK: }

func @multiple_islands(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    %island2:3 = tf_executor.island(%island1#2) {
      %sub = "tf.Sub"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %mul = "tf.Mul"(%sub, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %sub, %mul : tensor<*xi32>, tensor<*xi32>
    }
    %island3 = tf_executor.island {
      %sub = "tf.Sub"(%island1#0, %island2#0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %res = "tf.Print"(%sub) { message = "sub result" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield
    }
    tf_executor.fetch %island1#1, %island2#1, %island3 : tensor<*xi32>, tensor<*xi32>, !tf_executor.control
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @multiple_islands
// CHECK:  %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:    %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:    %[[SUB1:.*]], %[[SUB1_control:.*]] = tf_executor.island(%[[ADD2_control]]) wraps "tf.Sub"(%arg0, %arg1)
// CHECK:    %[[MUL:.*]], %[[MUL_control:.*]] = tf_executor.island(%[[SUB1_control]]) wraps "tf.Mul"(%[[SUB1]], %arg1)
// CHECK:    %[[SUB2:.*]], %[[SUB2_control:.*]] = tf_executor.island(%[[ADD2_control]], %[[MUL_control]]) wraps "tf.Sub"(%[[ADD1]], %[[SUB1]])
// CHECK:    %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island(%[[SUB2_control]]) wraps "tf.Print"(%[[SUB2]]) {message = "sub result"}
// CHECK:    tf_executor.fetch %[[ADD2]], %[[MUL]], %[[PRINT_control]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

func @dangling_print(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %graph:2 = tf_executor.graph {
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %res = "tf.Print"(%add2) { message = "add result" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add1, %add2 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island1#0, %island1#1 : tensor<*xi32>, tensor<*xi32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL:  func @dangling_print
// CHECK:  %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:    %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Add"(%[[ADD1_control:.*]], %arg1)
// CHECK:    %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island(%[[ADD2_control]]) wraps "tf.Print"(%[[ADD2_control:.*]]) {message = "add result"}
// CHECK:    tf_executor.fetch %[[ADD1]], %[[ADD2]], %[[PRINT_control]] :
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1

func @switch_and_merge(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
  %graph:2 = tf_executor.graph {
    %island0:3 = tf_executor.island {
      %add = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %less = "tf.Less"(%arg1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %res = "tf.Print"(%add) { message = "add result 1" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add, %less : tensor<*xi32>, tensor<i1>
    }
    %switch:3 = tf_executor.Switch %island0#0, %island0#1 : tensor<*xi32>
    %island1:2 = tf_executor.island {
      %add = "tf.Add"(%switch#0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %res = "tf.Print"(%add) { message = "add result 2" } : (tensor<*xi32>) -> (tensor<*xi32>)
      tf_executor.yield %add : tensor<*xi32>
    }
    %merge_out:3 = tf_executor.Merge %island1#0, %switch#1 : tensor<*xi32>
    tf_executor.fetch %merge_out#0, %merge_out#1 : tensor<*xi32>, tensor<i32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<i32>
}

// CHECK-LABEL:  func @switch_and_merge(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
// CHECK: %[[GRAPH:.*]]:2 = tf_executor.graph {
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[LESS:.*]], %[[LESS_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Less"(%arg1, %arg1)
// CHECK:   %[[PRINT1:.*]], %[[PRINT1_control:.*]] = tf_executor.island(%[[LESS_control]]) wraps "tf.Print"(%[[ADD1]]) {message = "add result 1"}
// CHECK:   %[[SWITCH_false:.*]], %[[SWITCH_true:.*]], {{.*}} = tf_executor.Switch %[[ADD1]], %[[LESS]], %[[PRINT1_control]]
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island wraps "tf.Add"(%[[SWITCH_false]], %arg1)
// CHECK:   %[[PRINT2:.*]], %[[PRINT2_control:.*]] = tf_executor.island(%[[ADD2_control]]) wraps "tf.Print"(%[[ADD2]]) {message = "add result 2"}
// CHECK:   %[[MERGE:.*]], %[[MERGE_index:.*]], %{{.*}} = tf_executor.Merge %[[ADD2]], %[[SWITCH_true]], %[[PRINT2_control]]
// CHECK:   tf_executor.fetch %[[MERGE]], %[[MERGE_index]]
// CHECK: }
// CHECK: return %[[GRAPH]]#0, %[[GRAPH]]#1

func @control_flow_plumbing(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %graph = tf_executor.graph {
    %island0:2 = tf_executor.island wraps "tf.Print"(%arg0) { message = "Random Print" } : (tensor<*xi32>) -> (tensor<*xi32>)
    %island1:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %add2, %island0#0 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island1#0 : tensor<*xi32>
  }
  return %graph : tensor<*xi32>
}

// CHECK-LABEL: func @control_flow_plumbing
// CHECK: %[[GRAPH:.*]] = tf_executor.graph {
// CHECK:   %[[PRINT:.*]], %[[PRINT_control:.*]] = tf_executor.island wraps "tf.Print"(%arg0) {message = "Random Print"}
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island(%[[PRINT_control]]) wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Add"(%[[ADD1]], %arg1)
// CHECK:   tf_executor.fetch %[[ADD2]] : tensor<*xi32>
// CHECK: }
// CHECK: return %[[GRAPH]] : tensor<*xi32>

func @fetching_arg(%arg0: tensor<*xi32>) {
  tf_executor.graph {
    %island:3 = tf_executor.island {
      %add1 = "tf.Add"(%arg0, %arg0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %add2 = "tf.Add"(%add1, %arg0) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      tf_executor.yield %arg0, %arg0 : tensor<*xi32>, tensor<*xi32>
    }
    tf_executor.fetch %island#2 : !tf_executor.control
  }
  return
}

// CHECK-LABEL: func @fetching_arg
// CHECK: tf_executor.graph {
// CHECK:   %[[ADD1:.*]], %[[ADD1_control:.*]] = tf_executor.island wraps "tf.Add"(%arg0, %arg0)
// CHECK:   %[[ADD2:.*]], %[[ADD2_control:.*]] = tf_executor.island(%[[ADD1_control]]) wraps "tf.Add"(%[[ADD1]], %arg0)
// CHECK:   tf_executor.fetch %[[ADD2_control]] : !tf_executor.control
// CHECK: }
