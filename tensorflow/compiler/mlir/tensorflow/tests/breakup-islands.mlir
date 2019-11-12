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
// CHECK:     %[[ADD1:.*]]:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:     %[[ADD2:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Add"(%[[ADD1]]#0, %arg1)
// CHECK:     tf_executor.fetch %[[ADD1]]#0, %[[ADD2]]#0 :
// CHECK:   }
// CHECK:   return %[[GRAPH]]#0, %[[GRAPH]]#1
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
// CHECK:    %[[ADD1:.*]]:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Add"(%[[ADD1]]#0, %arg1)
// CHECK:    %[[SUB1:.*]]:2 = tf_executor.island(%[[ADD2]]#1) wraps "tf.Sub"(%arg0, %arg1)
// CHECK:    %[[MUL:.*]]:2 = tf_executor.island(%[[SUB1]]#1) wraps "tf.Mul"(%[[SUB1]]#0, %arg1)
// CHECK:    %[[SUB2:.*]]:2 = tf_executor.island(%[[ADD2]]#1, %[[MUL]]#1) wraps "tf.Sub"(%[[ADD1]]#0, %[[SUB1]]#0)
// CHECK:    %[[PRINT:.*]]:2 = tf_executor.island(%[[SUB2]]#1) wraps "tf.Print"(%[[SUB2]]#0) {message = "sub result"}
// CHECK:    tf_executor.fetch %[[ADD2]]#0, %[[MUL]]#0, %[[PRINT]]#1 :
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
// CHECK:    %[[ADD1:.*]]:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:    %[[ADD2:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Add"(%1#0, %arg1)
// CHECK:    %[[PRINT:.*]]:2 = tf_executor.island(%[[ADD2]]#1) wraps "tf.Print"(%2#0) {message = "add result"}
// CHECK:    tf_executor.fetch %[[ADD1]]#0, %[[ADD2]]#0, %[[PRINT]]#1 :
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
// CHECK:   %[[ADD1:.*]]:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[LESS:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Less"(%arg1, %arg1)
// CHECK:   %[[PRINT1:.*]]:2 = tf_executor.island(%[[LESS]]#1) wraps "tf.Print"(%[[ADD1]]#0) {message = "add result 1"}
// CHECK:   %[[SWITCH:.*]]:3 = tf_executor.Switch %[[ADD1]]#0, %[[LESS]]#0, %[[PRINT1]]#1
// CHECK:   %[[ADD2:.*]]:2 = tf_executor.island wraps "tf.Add"(%[[SWITCH]]#0, %arg1)
// CHECK:   %[[PRINT2:.*]]:2 = tf_executor.island(%[[ADD2]]#1) wraps "tf.Print"(%[[ADD2]]#0) {message = "add result 2"}
// CHECK:   %[[MERGE:.*]]:3 = tf_executor.Merge %[[ADD2]]#0, %[[SWITCH]]#1, %[[PRINT2]]#1
// CHECK:   tf_executor.fetch %[[MERGE]]#0, %[[MERGE]]#1
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
// CHECK:   %[[PRINT:.*]]:2 = tf_executor.island wraps "tf.Print"(%arg0) {message = "Random Print"}
// CHECK:   %[[ADD1:.*]]:2 = tf_executor.island(%[[PRINT]]#1) wraps "tf.Add"(%arg0, %arg1)
// CHECK:   %[[ADD2:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Add"(%2#0, %arg1)
// CHECK:   tf_executor.fetch %[[ADD2]]#0 : tensor<*xi32>
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
// CHECK:   %[[ADD1:.*]]:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg0)
// CHECK:   %[[ADD2:.*]]:2 = tf_executor.island(%[[ADD1]]#1) wraps "tf.Add"(%[[ADD1]]#0, %arg0)
// CHECK:   tf_executor.fetch %[[ADD2]]#1 : !tf_executor.control
// CHECK: }
