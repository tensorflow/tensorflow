// RUN: mlir-hlo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @retain
func.func @retain(%arg0: memref<*xf32>, %arg1: memref<123xf32>)
    -> memref<456xf32> {
  %0 = deallocation.retain(%arg0) of(%arg1, %arg0)
      : (memref<*xf32>, memref<123xf32>, memref<*xf32>) -> memref<456xf32>
  return %0 : memref<456xf32>
}

// -----

func.func @invalid_retain(%arg0: memref<*xf32>, %arg1: memref<123xf64>)
    -> memref<456xf32> {
  // expected-error@+1 {{expected homogeneous operand and result element type}}
  %0 = deallocation.retain(%arg0) of(%arg1, %arg0)
      : (memref<*xf32>, memref<123xf64>, memref<*xf32>) -> memref<456xf32>
  return %0 : memref<456xf32>
}

// -----

func.func @invalid_retain_2(%arg0: memref<*xf32>, %arg1: memref<123xf32>)
    -> memref<456xf64> {
  // expected-error@+1 {{expected homogeneous operand and result element type}}
  %0 = deallocation.retain(%arg0) of(%arg1, %arg0)
      : (memref<*xf32>, memref<123xf32>, memref<*xf32>) -> memref<456xf64>
  return %0 : memref<456xf64>
}
