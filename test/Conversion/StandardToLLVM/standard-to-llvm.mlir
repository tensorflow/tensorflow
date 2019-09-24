// RUN: mlir-opt %s -lower-to-llvm | FileCheck %s

// CHECK-LABEL: func @address_space(
//       CHECK:   %{{.*}}: !llvm<"{ float addrspace(7)*, [1 x i64] }">)
func @address_space(%arg0 : memref<32xf32, (d0) -> (d0), 7>) {
  %0 = alloc() : memref<32xf32, (d0) -> (d0), 5>
  %1 = constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm<"float addrspace(5)*">
  %2 = load %0[%1] : memref<32xf32, (d0) -> (d0), 5>
  std.return
}

