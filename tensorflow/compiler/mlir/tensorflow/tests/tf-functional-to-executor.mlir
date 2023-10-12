// RUN: tf-opt -tf-functional-to-executor-conversion %s | FileCheck %s

func.func @multiple_return(%arg0 : tensor<*xi32>, %arg1 : tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %1 = "tf.Add"(%arg0, %arg1) {} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %2 = "tf.Add"(%1, %arg1) {} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %1, %2 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @multiple_return
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*xi32>, %[[ARG1:.*]]: tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>) {
// CHECK:   %[[GRAPH_RESULT:.*]]:2 = tf_executor.graph {
// CHECK:     %[[ISLAND_RESULT:.*]]:2, {{.*}} = tf_executor.island {
// CHECK:        %[[ADD1:.*]] = "tf.Add"(%[[ARG0]], %[[ARG1]]) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
// CHECK:        %[[ADD2:.*]] = "tf.Add"(%[[ADD1]], %[[ARG1]]) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
// CHECK:        tf_executor.yield %[[ADD1]], %[[ADD2]] : tensor<*xi32>, tensor<*xi32>
// CHECK:     }
// CHECK:     tf_executor.fetch %[[ISLAND_RESULT]]#0, %[[ISLAND_RESULT]]#1 : tensor<*xi32>, tensor<*xi32>
// CHECK:   }
// CHECK:   return %[[GRAPH_RESULT]]#0, %[[GRAPH_RESULT]]#1 : tensor<*xi32>, tensor<*xi32>

func.func @empty_graph() {
  func.return
}

// CHECK-LABEL: func @empty_graph
// CHECK: tf_executor.graph {
// CHECK:   %[[CONTROL:.*]] = tf_executor.island {
// CHECK:     tf_executor.yield
// CHECK:   }
// CHECK:   tf_executor.fetch %[[CONTROL]] : !tf_executor.control
// CHECK: }
// CHECK: return

func.func @graph_already() {
  tf_executor.graph {
    %control = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  func.return
}


// CHECK-LABEL: func @graph_already
// CHECK: tf_executor.graph {
// CHECK:   %[[CONTROL:.*]] = tf_executor.island {
// CHECK:     tf_executor.yield
// CHECK:   }
// CHECK:   tf_executor.fetch %[[CONTROL]] : !tf_executor.control
// CHECK: }
// CHECK: return

func.func @graph_and_more(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  tf_executor.graph {
    %control = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  %result = "tf.Add"(%arg0, %arg1) {} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %result : tensor<*xi32>
}

// CHECK-LABEL: func @graph_and_more
// CHECK:  %[[RESULT:.*]] = tf_executor.graph {
// CHECK:    %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK:      tf_executor.graph {
// CHECK:        %[[ISLAND_INNER:.*]] = tf_executor.island {
// CHECK:          tf_executor.yield
// CHECK:        }
// CHECK:        tf_executor.fetch %[[ISLAND_INNER]] : !tf_executor.control
// CHECK:      }
// CHECK:      %[[ADD:.*]] = "tf.Add"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
// CHECK:      tf_executor.yield %[[ADD]] : tensor<*xi32>
// CHECK:    }
// CHECK:    tf_executor.fetch %[[ISLAND]] : tensor<*xi32>
// CHECK:  }
// CHECK:  return %[[RESULT]] : tensor<*xi32>
