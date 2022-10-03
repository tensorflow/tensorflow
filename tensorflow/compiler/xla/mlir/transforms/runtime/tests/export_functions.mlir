// RUN: xla-runtime-opt %s --xla-rt-export-functions | FileCheck %s

// CHECK: func @single_result(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported} {
rt.export @single_result
func.func @single_result(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0 : memref<?xf32>
}

// CHECK: func @two_results(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported} {
rt.export @two_results
func.func @two_results(%arg0: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>) {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: rt.set_output %[[CTX]], 1, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0, %arg0 : memref<?xf32>, memref<?xf32>
}

// CHECK: func @not_exported(
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) -> memref<?xf32> {
func.func @not_exported(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NOT: rt.set_output
  // CHECK: return %[[ARG]]
  return %arg0 : memref<?xf32>
}
