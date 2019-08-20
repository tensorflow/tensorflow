// RUN: tf-opt -tf-executor-to-control-conversion %s  | FileCheck %s --check-prefix=CONTROL --dump-input=fail
// RUN: tf-opt -tf-control-to-executor-conversion %s  | FileCheck %s --check-prefix=EXECUTOR --dump-input=fail

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
