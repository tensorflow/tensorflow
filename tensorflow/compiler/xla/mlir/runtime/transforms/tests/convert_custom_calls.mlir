// RUN: xla-runtime-opt %s --xla-rt-convert-custom-calls | FileCheck %s

// CHECK-NOT: func private @custom_call(memref<?xf32>)
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = "target", attr0 = 1 : i32, attr1 = 1.0 : f32 }

// CHECK-NOT: func private @dynamic_custom_call(memref<?xf32>)
func.func private @dynamic_custom_call(%arg0: memref<?xf32>)
  attributes { rt.dynamic, rt.custom_call = "target" }

// CHECK: func @function_call_to_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) -> memref<?xf32> attributes {rt.exported = 0 : i32} {
func.func @function_call_to_custom_call(
  %arg0: !rt.execution_context,
  %arg1: memref<?xf32>
) -> memref<?xf32> attributes {rt.exported = 0 : i32} {
  // CHECK: %[[STATUS:.*]], %[[RES:.*]] = rt.call %[[CTX]]["target"]
  // CHECK-SAME: (%[[ARG]]) {attr0 = 2 : i32, attr1 = 1.000000e+00 : f32}
  // CHECK: %[[IS_OK:.*]] = rt.is_ok %[[STATUS]]
  // CHECK: assert %[[IS_OK]], "custom call 'target' failed"
  %0 = call @custom_call(%arg1) { attr0 = 2 : i32 }
       : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: func @function_call_to_dynamic_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) attributes {rt.exported = 0 : i32} {
func.func @function_call_to_dynamic_custom_call(
  %arg0: !rt.execution_context,
  %arg1: memref<?xf32>
) attributes {rt.exported = 0 : i32} {
  // CHECK: rt.call dynamic %[[CTX]]["target"]
  call @dynamic_custom_call(%arg1) : (memref<?xf32>) -> ()
  return
}

// CHECK: func @function_call_to_traced_custom_call(
// CHECK:   %[[CTX:.*]]: !rt.execution_context,
// CHECK:   %[[ARG:.*]]: memref<?xf32>
// CHECK: ) -> memref<?xf32> attributes {rt.exported = 0 : i32} {
func.func @function_call_to_traced_custom_call(
    %arg0: !rt.execution_context,
    %arg1: memref<?xf32>
) -> memref<?xf32> attributes {rt.exported = 0 : i32} {
  // CHECK: %[[RES:.*]]:2 = rt.trace #rt.hlo_trace<"fusion">, %[[CTX]]
  // CHECK-SAME: -> !rt.status, memref<?xf32> {
  // CHECK-NEXT:   %[[STATUS:.*]], %[[RET:.*]] = call %[[CTX]]["target"]
  // CHECK-NOT:    #rt.hlo_trace
  // CHECK-NEXT:   yield %[[STATUS]], %[[RET]] : !rt.status, memref<?xf32>
  // CHECK-NEXT: }
  // CHECK: rt.is_ok %[[RES]]#0
  %0 = call @custom_call(%arg1) { rt.trace = #rt.hlo_trace<"fusion"> }
    : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}