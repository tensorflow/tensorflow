// RUN: xla-runtime-opt %s --xla-rt-convert-to-entrypoint | FileCheck %s

// CHECK: func @single_result(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) {
func.func @single_result(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.entrypoint } {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0 : memref<?xf32>
}

// CHECK: func @two_results(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) {
func.func @two_results(%arg0: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>)
  attributes { rt.entrypoint } {
  // CHECK: rt.set_output %[[CTX]], 0, %[[ARG]] : memref<?xf32>
  // CHECK: rt.set_output %[[CTX]], 1, %[[ARG]] : memref<?xf32>
  // CHECK: return
  return %arg0, %arg0 : memref<?xf32>, memref<?xf32>
}

// CHECK: func @not_an_entrypoint(
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) -> memref<?xf32> {
func.func @not_an_entrypoint(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NOT: rt.set_output
  // CHECK: return %[[ARG]]
  return %arg0 : memref<?xf32>
}

// CHECK: func @assert_to_error(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ASSERT:.*]]: i1
// CHECK: ) {
func.func @assert_to_error(%arg0: i1) attributes { rt.entrypoint } {
  // CHECK: cond_br %[[ASSERT]], ^[[OK:.*]], ^[[ERR:.*]]
  // CHECK: ^[[OK]]:
  // CHECK:   return
  // CHECK: ^[[ERR]]:
  // CHECK:   rt.set_error %[[CTX]], "Failed precondition"
  // CHECK:   return
  cf.assert %arg0, "Failed precondition"
  return
}

// Custom call prototype declaration.
// CHECK-NOT: func private @custom_call(memref<?xf32>)
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = "target", attr0 = 1 : i32, attr1 = 1.0 : f32 }

// CHECK: func @function_call_to_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// )
func.func @function_call_to_custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.entrypoint } {
  // CHECK: %[[STATUS:.*]], %[[RES:.*]] = rt.custom_call %[[CTX]]["target"]
  // CHECK-SAME: (%[[ARG]]) {attr0 = 2 : i32, attr1 = 1.000000e+00 : f32}
  // CHECK: %[[IS_OK:.*]] = rt.is_ok %[[STATUS]]
  // CHECK: cf.cond_br %[[IS_OK]], ^[[OK:.*]], ^[[ERR:.*]]
  // CHECK: ^[[OK]]:
  // CHECK:   rt.set_output %[[CTX]], 0, %[[RES]] : memref<?xf32>
  // CHECK: ^[[ERR]]:
  // CHECK:   rt.set_error %arg0, "custom call 'target' failed"
  %0 = call @custom_call(%arg0) { attr0 = 2 : i32 }
       : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// Direct custom call prototype declaration.
// CHECK-NOT: func private @direct_custom_call(memref<?xf32>)
func.func private @direct_custom_call(%arg0: memref<?xf32>)
  attributes { rt.direct_custom_call = "target" }

// CHECK: func @function_call_to_direct_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// )
func.func @function_call_to_direct_custom_call(%arg0: memref<?xf32>)
  attributes { rt.entrypoint } {
  // CHECK: rt.custom_call direct %[[CTX]]["target"]
  call @direct_custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}
