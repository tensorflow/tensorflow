// RUN: xla-runtime-opt %s --xla-runtime-default-cpu-pipeline | FileCheck %s
// RUN: xla-runtime-opt %s --xla-runtime-default-gpu-pipeline | FileCheck %s

// Check that entrypoint function was lowered to LLVM function with expected
// ABI.

// CHECK-LABEL: llvm.func @main(
// CHECK-SAME:    %[[ARG0:arg[0-9]+]]: !llvm.ptr,
// CHECK-SAME:    %[[ARG1:arg[0-9]+]]: !llvm.ptr<f32>,
// CHECK-SAME:    %[[ARG2:arg[0-9]+]]: !llvm.ptr<f32>,
// CHECK-SAME:    %[[ARG3:arg[0-9]+]]: i64,
// CHECK-SAME:    %[[ARG4:arg[0-9]+]]: i64,
// CHECK-SAME:    %[[ARG5:arg[0-9]+]]: i64
// CHECK-SAME:  )
rt.export @main ordinal 0
func.func @main(%arg0: memref<?xf32>) {
  call @custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}

// Check that XLA runtime custom call was lowered to a LLVM function call.

// CHECK: llvm.func @target
// CHECK-SAME: passthrough = ["nounwind"]
func.func private @custom_call(%arg0: memref<?xf32>)
  attributes { rt.custom_call = "target" }
