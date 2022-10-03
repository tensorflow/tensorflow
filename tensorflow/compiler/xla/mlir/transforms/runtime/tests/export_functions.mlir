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

// Custom call prototype declaration.
// CHECK-NOT: func private @custom_call(memref<?xf32>)
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = "target", attr0 = 1 : i32, attr1 = 1.0 : f32 }

// CHECK: func @function_call_to_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported} {
rt.export @function_call_to_custom_call
func.func @function_call_to_custom_call(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK: %[[STATUS:.*]], %[[RES:.*]] = rt.custom_call %[[CTX]]["target"]
  // CHECK-SAME: (%[[ARG]]) {attr0 = 2 : i32, attr1 = 1.000000e+00 : f32}
  // CHECK: %[[IS_OK:.*]] = rt.is_ok %[[STATUS]]
  // CHECK: assert %[[IS_OK]], "custom call 'target' failed"
  %0 = call @custom_call(%arg0) { attr0 = 2 : i32 }
       : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// Dynamic custom call prototype declaration.
// CHECK-NOT: func private @dynamic_custom_call(memref<?xf32>)
func.func private @dynamic_custom_call(%arg0: memref<?xf32>)
  attributes { rt.dynamic, rt.custom_call = "target" }

// CHECK: func @function_call_to_dynamic_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported} {
rt.export @function_call_to_dynamic_custom_call
func.func @function_call_to_dynamic_custom_call(%arg0: memref<?xf32>) {
  // CHECK: rt.custom_call dynamic %[[CTX]]["target"]
  call @dynamic_custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}

// CHECK: func @function_call_to_traced_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported} {
rt.export @function_call_to_traced_custom_call
func.func @function_call_to_traced_custom_call(%arg0: memref<?xf32>)
    -> memref<?xf32> {
  // CHECK: %[[RES:.*]]:2 = rt.trace #rt.hlo_trace<"fusion", "foo", 0>, %[[CTX]]
  // CHECK-SAME: -> !rt.status, memref<?xf32> {
  // CHECK-NEXT:   %[[STATUS:.*]], %[[RET:.*]] = custom_call %[[CTX]]["target"]
  // CHECK-NOT:    #rt.hlo_trace
  // CHECK-NEXT:   yield %[[STATUS]], %[[RET]] : !rt.status, memref<?xf32>
  // CHECK-NEXT: }
  // CHECK: rt.is_ok %[[RES]]#0
  %0 = call @custom_call(%arg0) { rt.trace = #rt.hlo_trace<"fusion", "foo", 0> }
    : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}
