// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @store_vector_memref() -> memref<1x2xvector<2xi32>> {
  %m = memref.alloc() : memref<1x2xvector<2xi32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = arith.constant dense<[1,2]> : vector<2xi32>
  vector.store %v, %m[%c0, %c1] : memref<1x2xvector<2xi32>>, vector<2xi32>
  return %m : memref<1x2xvector<2xi32>>
}

// CHECK-LABEL: @store_vector_memref
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<1x2xvector<2xi32>>: [[[0, 0], [1, 2]]]

func.func @store_scalar_memref() -> memref<2x2xi32> {
  %m = memref.alloc() : memref<2x2xi32>
  %v = arith.constant dense<[[1,2]]> : vector<1x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  vector.store %v, %m[%c1, %c0] : memref<2x2xi32>, vector<1x2xi32>
  return %m : memref<2x2xi32>
}

// CHECK-LABEL: @store_scalar_memref
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<2x2xi32>: [[0, 0], [1, 2]]

func.func @store_oob() -> memref<2x2xi32> {
  %m = memref.alloc() : memref<2x2xi32>
  %v = arith.constant dense<[[1,2]]> : vector<1x2xi32>
  %c1 = arith.constant 1 : index
  vector.store %v, %m[%c1, %c1] : memref<2x2xi32>, vector<1x2xi32>
  return %m : memref<2x2xi32>
}

// CHECK-LABEL: @store_oob
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<2x2xi32>: [[0, 0], [0, 1]]
