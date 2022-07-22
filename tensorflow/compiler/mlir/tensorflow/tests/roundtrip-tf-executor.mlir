// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=MlirRoundtripPass | FileCheck %s

// The test uses the tf_graph_optimization_pass to run the MlirRoundtripPass.
// We convert mlir -> Graph -> mlir -> Graph -> mlir

func.func @main() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf.NoOp"() {} : () -> () loc("X")
    tf_executor.fetch
  }
  func.return
}

// Check for the presence of tf.NoOp in the final output.
// CHECK: tf.NoOp
