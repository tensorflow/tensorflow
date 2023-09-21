// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @no_inputs() -> tensor<4xf32> {
  %init = arith.constant dense<[1.0,2.0,3.0,4.0]> : tensor<4xf32>
  %zero = linalg.map outs(%init:tensor<4xf32>)() {
    %0 = arith.constant 0.0: f32
    linalg.yield %0: f32
  }
  func.return %zero : tensor<4xf32>
}

// CHECK-LABEL: @no_inputs
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]

func.func @binary() -> tensor<4xi32> {
  %init = tensor.empty() : tensor<4xi32>
  %lhs = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %rhs = arith.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
  %add = linalg.map ins(%lhs, %rhs: tensor<4xi32>, tensor<4xi32>)
                    outs(%init: tensor<4xi32>)
    (%lhs_elem: i32, %rhs_elem: i32) {
      %0 = arith.addi %lhs_elem, %rhs_elem: i32
      linalg.yield %0: i32
    }
  func.return %add : tensor<4xi32>
}

// CHECK-LABEL: @binary
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [10, 21, 32, 43]

func.func @memref() -> memref<4xi32> {
  %alloc = memref.alloc() : memref<4xi32>
  %lhs = arith.constant dense<[0, 1, 2, 3]> : memref<4xi32>
  %rhs = arith.constant dense<[10, 20, 30, 40]> : memref<4xi32>
  linalg.map ins(%lhs, %rhs: memref<4xi32>, memref<4xi32>)
             outs(%alloc: memref<4xi32>)
    (%lhs_elem: i32, %rhs_elem: i32) {
      %0 = arith.muli %lhs_elem, %rhs_elem: i32
      linalg.yield %0: i32
    }
  func.return %alloc : memref<4xi32>
}

// CHECK-LABEL: @memref
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 20, 60, 120]

func.func @index() -> memref<4xindex> {
  %alloc = memref.alloc() : memref<4xindex>
  linalg.map outs(%alloc: memref<4xindex>)() {
    %0 = linalg.index 0 : index
    linalg.yield %0: index
  }
  func.return %alloc : memref<4xindex>
}

// CHECK-LABEL: @index
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 1, 2, 3]

func.func @vector() -> memref<4xvector<2xindex>> {
  %c = arith.constant dense<42> : vector<2xindex>
  %alloc = memref.alloc() : memref<4xvector<2xindex>>
  linalg.map outs(%alloc: memref<4xvector<2xindex>>)() {
    linalg.yield %c: vector<2xindex>
  }
  func.return %alloc : memref<4xvector<2xindex>>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[42, 42], [42, 42], [42, 42], [42, 42]]
