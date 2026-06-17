// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

memref.global "private" constant @cst : memref<2xi16> = dense<[1, 2]>

func.func @get_global() -> memref<2xi16> {
  %0 = memref.get_global @cst : memref<2xi16>
  return %0 : memref<2xi16>
}

// CHECK-LABEL: @get_global
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 2]
