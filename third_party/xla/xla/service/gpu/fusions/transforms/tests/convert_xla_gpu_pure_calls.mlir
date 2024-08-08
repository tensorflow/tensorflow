// RUN: mlir_fusions_opt %s -xla-gpu-convert-pure-call-ops | FileCheck %s
// RUN: mlir_fusions_opt %s -cse -xla-gpu-convert-pure-call-ops \
// RUN: | FileCheck %s -check-prefixes=CHECK-CSE

func.func private @callee() -> f32 {
  %ret = arith.constant 0.0 : f32
  return %ret : f32
}

func.func @caller() -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %call0 = xla_gpu.pure_call @callee() : () -> (f32)
  %v = scf.for %i = %c0 to %c10 step %c1 iter_args(%r = %call0) -> f32 {
    %call1 = xla_gpu.pure_call @callee() : () -> (f32)
    %new_v = arith.addf %call1, %r : f32
    scf.yield %new_v : f32
  }
  return %v : f32
}

// CHECK-LABEL: @caller
// CHECK:   call @callee
// CHECK:   call @callee

// CHECK-CSE: @caller
// CHECK-CSE: %[[CALL:.*]] = call @callee
// CHECK-CSE: scf.for {{.*}} iter_args(%[[ITER_ARG:.*]] = %[[CALL]])
// CHECK-CSE: arith.addf %[[CALL]], %[[ITER_ARG]]

// -----

func.func private @arg_callee(%arg0: f32, %arg1: f32) -> f32 {
  %ret = arith.addf %arg0, %arg1 : f32
  return %ret : f32
}

func.func @arg_caller() -> f32 {
  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %call = xla_gpu.pure_call @arg_callee(%cst0, %cst1) : (f32, f32) -> (f32)
  return %call : f32
}

// CHECK-LABEL: @arg_caller
// CHECK: %[[CST0:.*]] = arith.constant 0
// CHECK: %[[CST1:.*]] = arith.constant 1
// CHECK: %[[RET:.*]] = call @arg_callee(%[[CST0]], %[[CST1]])
// CHECK: return %[[RET]]
