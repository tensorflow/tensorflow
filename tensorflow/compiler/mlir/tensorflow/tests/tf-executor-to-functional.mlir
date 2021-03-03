// RUN: tf-opt -tf-executor-to-functional-conversion %s -split-input-file -verify-diagnostics | FileCheck %s

func @unsupported_op() {
  tf_executor.graph {
    // expected-error@+1 {{'tf_executor.ControlTrigger' op is not supported for lifting out of tf_executor.graph, expected tf_executor.island}}
    %control = tf_executor.ControlTrigger {}
    tf_executor.fetch
  }
  return
}

// -----

// CHECK-LABEL: func @empty_graph
// CHECK-NEXT: return
func @empty_graph() {
  tf_executor.graph {
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @empty_island
// CHECK-NEXT: return
func @empty_island() {
  tf_executor.graph {
    %control = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @island_forwarding_result
// CHECK-SAME: ([[ARG_0:%.*]]: tensor<i32>)
// CHECK-NEXT: return [[ARG_0]]
func @island_forwarding_result(%arg0: tensor<i32>) -> tensor<i32> {
  %graph_result = tf_executor.graph {
    %output, %control = tf_executor.island {
      tf_executor.yield %arg0 : tensor<i32>
    }
    tf_executor.fetch %output : tensor<i32>
  }
  return %graph_result : tensor<i32>
}

// CHECK-LABEL: func @transitive_data_dependencies
// CHECK-SAME: ([[ARG_0:%.*]]: tensor<i32>)
// CHECK-NEXT: [[A:%.*]] = "tf.opA"([[ARG_0]])
// CHECK-NEXT: [[B:%.*]] = "tf.opB"([[A]])
// CHECK-NEXT: return [[B]]
func @transitive_data_dependencies(%arg0: tensor<i32>) -> tensor<i32> {
  %graph_result = tf_executor.graph {
    %output0, %control0 = tf_executor.island {
      %a = "tf.opA"(%arg0) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %a : tensor<i32>
    }
    %output1, %control1 = tf_executor.island {
      %b = "tf.opB"(%output0) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %b : tensor<i32>
    }
    tf_executor.fetch %output1 : tensor<i32>
  }
  return %graph_result : tensor<i32>
}

// CHECK-LABEL: func @transitive_control_dependencies
// CHECK-SAME: ([[ARG_0:%.*]]: tensor<i32>)
// CHECK-NEXT: "tf.opA"([[ARG_0]])
// CHECK-NEXT: [[B:%.*]] = "tf.opB"([[ARG_0]])
// CHECK-NEXT: return [[B]]
func @transitive_control_dependencies(%arg0: tensor<i32>) -> tensor<i32> {
  %graph_result = tf_executor.graph {
    %output0, %control0 = tf_executor.island {
      %a = "tf.opA"(%arg0) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %a : tensor<i32>
    }
    %output1, %control1 = tf_executor.island(%control0) {
      %b = "tf.opB"(%arg0) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %b : tensor<i32>
    }
    tf_executor.fetch %output1 : tensor<i32>
  }
  return %graph_result : tensor<i32>
}

// CHECK-LABEL: func @multiple_inner_ops
// CHECK-SAME: ([[ARG_0:%.*]]: tensor<i32>)
// CHECK-NEXT: [[A:%.*]] = "tf.opA"([[ARG_0]])
// CHECK-NEXT: [[B:%.*]] = "tf.opB"([[A]])
// CHECK-NEXT: [[C:%.*]] = "tf.opC"([[A]])
// CHECK-NEXT: [[D:%.*]] = "tf.opD"([[C]])
// CHECK-NEXT: return [[B]], [[D]]
func @multiple_inner_ops(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %graph_result:2 = tf_executor.graph {
    %output0_0, %output0_1, %control0 = tf_executor.island {
      %a = "tf.opA"(%arg0) : (tensor<i32>) -> tensor<i32>
      %b = "tf.opB"(%a) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %a, %b : tensor<i32>, tensor<i32>
    }
    %output1_0, %output1_1, %control1 = tf_executor.island {
      %c = "tf.opC"(%output0_0) : (tensor<i32>) -> tensor<i32>
      %d = "tf.opD"(%c) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %c, %d : tensor<i32>, tensor<i32>
    }
    tf_executor.fetch %output0_1, %output1_1 : tensor<i32>, tensor<i32>
  }
  return %graph_result#0, %graph_result#1 : tensor<i32>, tensor<i32>
}
