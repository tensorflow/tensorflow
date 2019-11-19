// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=MlirRoundtripPass | FileCheck %s --dump-input-on-failure

module {
  func @main() {
    tf_executor.graph {
      %0 = tf_executor.island {
        "tf.NoOp"() {} : () -> () loc("X")
        tf_executor.yield
      }
      tf_executor.fetch
    }
    return
  }
}

// The test uses the tf_graph_optimization_pass to run the MlirRoundtripPass.
// We convert mlir -> Graph -> mlir -> Graph -> mlir
// Check for the presence of tf.NoOp in the final output.
// CHECK: tf.NoOp
