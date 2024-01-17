// RUN: xla-gpu-opt %s -xla-add-hlo-trace-annotations | FileCheck %s

module attributes { mhlo.unique_id = 42 : i64 } {

func.func private @local_xla.foo() attributes { rt.custom_call = "xla.foo" }

// CHECK: func @func() {
func.func @func() {
  // CHECK: call @local_xla.foo()
  // CHECK-SAME: rt.trace = #rt.hlo_trace<"gemm.name.42">
  call @local_xla.foo() : () -> () loc("gemm.name.42")
  return
}

} loc("module-name")
