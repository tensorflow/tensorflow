// RUN: mlir-hlo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @retain
func.func @retain(%arg0: memref<2xf32>, %arg1: !deallocation.ownership, %arg2: !deallocation.ownership)
    -> !deallocation.ownership {
  %0 = deallocation.retain(%arg0) of(%arg1, %arg2)
      : (memref<2xf32>, !deallocation.ownership, !deallocation.ownership) -> !deallocation.ownership
  return %0 : !deallocation.ownership
}

// CHECK-LABEL: @get_buffer
func.func @get_buffer(%arg0: memref<2xf32>) -> index {
  %0 = deallocation.get_buffer %arg0 : memref<2xf32>
  return %0 : index
}

// CHECK-LABEL: @get_ownership_buffer
func.func @get_ownership_buffer(%arg0: !deallocation.ownership) -> index {
  %0 = deallocation.get_buffer %arg0 : !deallocation.ownership
  return %0 : index
}

// CHECK-LABEL: @own
func.func @own(%arg0: memref<2xf32>) -> !deallocation.ownership {
  %0 = deallocation.own %arg0 : memref<2xf32>
  return %0 : !deallocation.ownership
}

// CHECK-LABEL: @null
func.func @null() -> !deallocation.ownership {
  %0 = deallocation.null
  return %0 : !deallocation.ownership
}

// CHECK-LABEL: @free
func.func @free(%arg0: !deallocation.ownership) {
  deallocation.free %arg0
  return
}