// RUN: not mlir-bisect %s \
// RUN: --pass-pipeline="builtin.module(test-break-linalg-transpose)" \
// RUN: | FileCheck %s

func.func @main() -> memref<2x2xindex> {
  %a = memref.alloc() : memref<2x2xindex>
  return %a : memref<2x2xindex>
}

// CHECK: Did not find bug in initial module
