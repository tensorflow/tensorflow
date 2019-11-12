// RUN: mlir-opt %s -lower-to-llvm -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @address_space(
//       CHECK:   %{{.*}}: !llvm<"{ float addrspace(7)*, float addrspace(7)*, i64, [1 x i64], [1 x i64] }*">)
//       CHECK:   llvm.load %{{.*}} : !llvm<"{ float addrspace(7)*, float addrspace(7)*, i64, [1 x i64], [1 x i64] }*">
func @address_space(%arg0 : memref<32xf32, (d0) -> (d0), 7>) {
  %0 = alloc() : memref<32xf32, (d0) -> (d0), 5>
  %1 = constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm<"float addrspace(5)*">
  %2 = load %0[%1] : memref<32xf32, (d0) -> (d0), 5>
  std.return
}

// CHECK-LABEL: func @strided_memref(
func @strided_memref(%ind: index) {
  %0 = alloc()[%ind] : memref<32x64xf32, (i, j)[M] -> (32 + M * i + j)>
  std.return
}

// -----

// This should not crash. The first operation cannot be converted, so the
// secound should not match. This attempts to convert `return` to `llvm.return`
// and complains about non-LLVM types.
func @unknown_source() -> i32 {
  %0 = "foo"() : () -> i32
  %1 = addi %0, %0 : i32
  // expected-error@+1 {{must be LLVM dialect type}}
  return %1 : i32
}
