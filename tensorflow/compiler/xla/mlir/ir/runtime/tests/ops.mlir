// RUN: xla-runtime-opt %s | FileCheck %s

// CHECK-LABEL: func @pass_context(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @pass_context(%arg0: !rt.execution_context) {
  return
}

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
  // CHECK: rt.custom_call %[[CTX]]["f32_reduce"] (%[[MEMREF]])
  // CHECK-SAME: : (memref<?xf32>) -> f32
  %status, %0 = rt.custom_call %ctx["f32_reduce"] (%input)
                : (memref<?xf32>) -> f32
  %ok = rt.is_ok %status
  cf.assert %ok, "failed to call custom call"
  return %0 : f32
}

// CHECK-LABEL: func @direct_custom_call(
// CHECK:  %[[CTX:.*]]: !rt.execution_context
func.func @direct_custom_call(%ctx: !rt.execution_context) {
  // CHECK: rt.custom_call direct %[[CTX]]["f32_reduce"] () : () -> ()
  %status = rt.custom_call direct %ctx["f32_reduce"] () : () -> ()
  return
}
