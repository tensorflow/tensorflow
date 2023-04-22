// RUN: tf-opt -tf-executor-graph-pruning %s  | FileCheck %s --check-prefix=CONTROL

// CONTROL-LABEL: func @main
// CONTROL-NEXT:    return

// EXECUTOR-LABEL: func @main
// EXECUTOR-NEXT:    tf_executor.graph {
// EXECUTOR-NEXT:      tf_executor.fetch
// EXECUTOR-NEXT:    }
// EXECUTOR-NEXT:    return

func @main() {
  return
}
