// RUN: tf-opt %s -xla-legalize-tf -split-input-file -verify-diagnostics

func @tf_executor_graph_op() {
    // expected-error@+1 {{failed to legalize operation 'tf_executor.graph'}}
    tf_executor.graph {
      %0 = tf_executor.island {
        "tf.NoOp"() {} : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    return

}

// -----

func @tf_unknown_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+1 {{failed to legalize operation 'tf.OpA'}}
  %0 = "tf.OpA"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

func @tf_known_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tf.Add"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}
