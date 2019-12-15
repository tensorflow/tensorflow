// RUN: tf-opt %s -tf-executor-graph-pruning=skip-main-func | FileCheck %s --dump-input=fail

// Check that @main function is skipped by default.
// CHECK-LABEL: func @main
func @main() {
  tf_executor.graph {
    // CHECKT: tf_executor.island
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}
