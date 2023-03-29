// RUN: mlir-hlo-opt -hlo-convert-deallocation-ops-to-llvm %s \
// RUN: -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @null()
func.func @null() -> !deallocation.ownership {
  %null = deallocation.null
  func.return %null : !deallocation.ownership
}
// CHECK: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK: %[[RET:.*]] = builtin.unrealized_conversion_cast %[[NULL]]
// CHECK: return %[[RET]]

// -----

// CHECK-LABEL: func.func @memref_get_buffer
func.func @memref_get_buffer(%arg0: memref<2x?xf32>) -> index {
  %ret = deallocation.get_buffer %arg0 : memref<2x?xf32>
  return %ret : index
}

// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue
// CHECK-NEXT: llvm.ptrtoint

// -----

// CHECK-LABEL: func.func @ownership_get_buffer
func.func @ownership_get_buffer(%arg0: !deallocation.ownership) -> index {
  %ret = deallocation.get_buffer %arg0 : !deallocation.ownership
  return %ret : index
}

// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.ptrtoint

// -----

// CHECK-LABEL: func.func @own(
func.func @own(%arg0: memref<2x?xf32>) -> !deallocation.ownership {
  %ret = deallocation.own %arg0 : memref<2x?xf32>
  return %ret : !deallocation.ownership
}

// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue
// CHECK-NEXT: builtin.unrealized_conversion_cast

// -----

func.func @freeAlloc(%arg0: !deallocation.ownership) {
  deallocation.free %arg0
  return
}

// CHECK: @freeAlloc
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.call @free
