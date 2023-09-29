// RUN: xla-gpu2-opt %s -split-input-file | FileCheck %s

func.func @main(%arg0: !xla_gpu.execution_context) {
  return
}

// CHECK-LABEL: func @main(
// CHECK:   %[[ARG0:.+]]: !xla_gpu.execution_context
// CHECK: ) {
// CHECK:   return
// CHECK: }
