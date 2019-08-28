// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=MlirRoundtripPass | FileCheck %s --dump-input-on-failure

// The test uses the tf_graph_optimization_pass to run the MlirRoundtripPass.
// We convert mlir -> Graph -> mlir -> Graph -> mlir

func @main() {
  "_tf.NoOp"() {} : () -> () loc("X")
  return
}

// Check for the presence of tf.NoOp in the final output.
// CHECK: tf.NoOp