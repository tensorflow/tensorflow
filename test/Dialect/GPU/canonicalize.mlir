// RUN: mlir-opt -pass-pipeline='func(canonicalize)' %s | FileCheck %s

// CHECK-LABEL: @propagate_constant
// CHECK-SAME:  %[[arg1:.*]]: memref
func @propagate_constant(%arg1: memref<?xf32>) {
  // The outer constant must be preserved because it still has uses.
  // CHECK: %[[outer_cst:.*]] = constant 1
  %c1 = constant 1 : index

  // The constant must be dropped from the args list, but the memref should
  // remain.
  // CHECK: gpu.launch
  // CHECK-SAME: args(%[[inner_arg:.*]] = %[[arg1]]) : memref
  gpu.launch blocks(%bx, %by, %bz) in (%sbx = %c1, %sby = %c1, %sbz = %c1)
             threads(%tx, %ty, %tz) in (%stx = %c1, %sty = %c1, %stz = %c1)
             args(%x = %c1, %y = %arg1) : index, memref<?xf32> {
    // The constant is propagated into the kernel body and used.
    // CHECK: %[[inner_cst:.*]] = constant 1
    // CHECK: "foo"(%[[inner_cst]])
    "foo"(%x) : (index) -> ()

    // CHECK: "bar"(%[[inner_arg]])
    "bar"(%y) : (memref<?xf32>) -> ()
    gpu.return
  }
  return
}

