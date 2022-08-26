// RUN: mlir-hlo-opt --alloc-to-arg %s -verify-diagnostics -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func @alloc_to_arg
// CHECK-SAME: (%arg0: memref<8xf32>, %arg1: memref<8xf32> {my.attr})
func.func @alloc_to_arg(%arg0: memref<8xf32>) -> (memref<8xf32> {my.attr}) {
  // CHECK-NOT: memref.alloc
  %0 = memref.alloc() : memref<8xf32>
  return %0 : memref<8xf32>
}

// -----

func.func @not_alloc(%arg0: memref<8xf32>) -> memref<8xf32> {
  // expected-error@+1 {{expected operand #0 to be defined by an memref.alloc}}
  return %arg0 : memref<8xf32>
}
