// RUN: xla-translate-opt %s -split-input-file -outline-with-xla-framework | FileCheck %s

// CHECK-LABEL: @main_xla_framework
// CHECK-SAME: %[[ARG0:.*]]: !xla_framework.buffer
// CHECK-SAME: -> !xla_framework.buffer
// CHECK-SAME: attributes {xla_entry = true}
// CHECK-NEXT: %[[BUF0:.*]] = xla_framework.buffer_to_mem %[[ARG0]] : memref<?xf32>
// CHECK-NEXT: %[[BUF1:.*]] = call @main(%[[BUF0]])
// CHECK-NEXT: %[[RESULT:.*]] = xla_framework.mem_to_buffer %[[BUF1]] : memref<?xf32>
// CHECK-NEXT: return %[[RESULT]] : !xla_framework.buffer
func.func @main(%arg0: memref<?xf32>) -> memref<?xf32> {
  func.return %arg0 : memref<?xf32>
}

// CHECK: func private @main(%arg0: memref<?xf32>) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<internal>}
