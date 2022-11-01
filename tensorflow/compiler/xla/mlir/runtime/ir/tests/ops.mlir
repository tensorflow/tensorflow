// RUN: xla-runtime-opt %s | FileCheck %s

// CHECK: rt.export @pass_context
rt.export @pass_context

// CHECK-LABEL: func @pass_context(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @pass_context(%arg0: !rt.execution_context) {
  return
}

// CHECK: rt.export @set_output ordinal 42
rt.export @set_output ordinal 42

// CHECK-LABEL: func @set_output(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @set_output(%arg0: !rt.execution_context) {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %0 = memref.alloc() : memref<f32>
  // CHECK: rt.set_output %[[CTX]], 0, %[[MEMREF]]
  rt.set_output %arg0, 0, %0 : memref<f32>
  return
}

// CHECK-LABEL: func @set_error(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @set_error(%arg0: !rt.execution_context) {
  // CHECK: rt.set_error %[[CTX]], "Failed precondition"
  rt.set_error %arg0, "Failed precondition"
  return
}

// CHECK-LABEL: func @custom_call(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
// CHECK:  %[[MEMREF:.*]]: memref<?xf32>
func.func @custom_call(%ctx: !rt.execution_context,
                       %input: memref<?xf32>) -> f32 {
  // CHECK: rt.call %[[CTX]]["f32_reduce"] (%[[MEMREF]])
  // CHECK-SAME: : (memref<?xf32>) -> f32
  %status, %0 = rt.call %ctx["f32_reduce"] (%input) : (memref<?xf32>) -> f32
  %ok = rt.is_ok %status
  cf.assert %ok, "failed to call custom call"
  return %0 : f32
}

// CHECK-LABEL: func @dynamic_custom_call(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @dynamic_custom_call(%ctx: !rt.execution_context) {
  // CHECK: rt.call dynamic %[[CTX]]["f32_reduce"] () : () -> ()
  %status = rt.call dynamic %ctx["f32_reduce"] () : () -> ()
  return
}

// CHECK-LABEL: func @opaque_arg(
// CHECK:  %[[CTX:.*]]: !rt.execution_context,
// CHECK:  %[[ARG:.*]]: !rt.opaque
// CHECK: ) -> !rt.opaque
func.func @opaque_arg(%ctx: !rt.execution_context,
                      %arg0: !rt.opaque) -> !rt.opaque {
  // CHECK: rt.call %[[CTX]]["test"]
  // CHECK-SAME: (%[[ARG]]) : (!rt.opaque) -> !rt.opaque
  %status, %result = rt.call %ctx["test"] (%arg0) : (!rt.opaque) -> (!rt.opaque)
  return %result : !rt.opaque
}

// CHECK-LABEL: func @trace(
// CHECK:  %[[CTX:.*]]: !rt.execution_context,
// CHECK:  %[[ARG:.*]]: memref<?x?xf32>
// CHECK: ) -> memref<?x?xf32>
func.func @trace(%ctx: !rt.execution_context,
                 %arg: memref<?x?xf32>) -> memref<?x?xf32> {
  // CHECK: rt.trace #rt.hlo_trace<"fusion", "foo", 0>, %[[CTX]]
  rt.trace #rt.hlo_trace<"fusion", "foo", 0>, %ctx {}

  // CHECK: rt.trace #rt.hlo_trace<"fusion", "bar", 0>
  // CHECK-SAME: %[[CTX]] -> memref<?x?xf32>
  // CHECK-NEXT: yield %[[ARG]] : memref<?x?xf32>
  %0 = rt.trace #rt.hlo_trace<"fusion", "bar", 0>, %ctx -> memref<?x?xf32> {
    yield %arg : memref<?x?xf32>
  }

  return %0 : memref<?x?xf32>
}
