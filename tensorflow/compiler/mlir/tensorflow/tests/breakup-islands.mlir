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
// CHECK:     tf_executor.fetch %[[ADD1]]#0, %[[ADD2]]#0
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
// CHECK:    %[[SUB2:.*]]:2 = tf_executor.island wraps "tf.Sub"(%[[ADD1]]#0, %[[SUB1]]#0)
// CHECK:    %[[PRINT:.*]]:2 = tf_executor.island(%[[SUB2]]#1) wraps "tf.Print"(%[[SUB2]]#0) {message = "sub result"}
// CHECK:    tf_executor.fetch %[[ADD2]]#0, %[[MUL]]#0, %[[PRINT]]#1
// CHECK:  }
// CHECK:  return %[[GRAPH]]#0, %[[GRAPH]]#1
