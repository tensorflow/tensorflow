// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bounds_check() -> tensor<10xi32> {
  %operand = arith.constant dense<0> : tensor<10xi32>
  %indices = arith.constant dense<[[1], [8], [-1]]> : tensor<3x1xindex>
  %updates = arith.constant dense<[[4, 5, 6], [6, 7, 8], [8, 9, 10]]> : tensor<3x3xi32>
  %scatter = thlo.scatter ins(%indices: tensor<3x1xindex>, %updates: tensor<3x3xi32>)
                          outs(%operand: tensor<10xi32>) (%in: i32, %out: i32) {
    %add = arith.addi %in, %out : i32
    thlo.yield %add : i32
  }
  return %scatter : tensor<10xi32>
}

// CHECK-LABEL: @bounds_check
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 4, 5, 6, 0, 0, 0, 0, 0, 0]

func.func @update_last_element() -> tensor<2xi32> {
  %operand = arith.constant dense<[1, 1]> : tensor<2xi32>
  %indices = arith.constant dense<[[1]]> : tensor<1x1xindex>
  %updates = arith.constant dense<[[0]]> : tensor<1x1xi32>

  %scatter = thlo.scatter ins(%indices: tensor<1x1xindex>, %updates: tensor<1x1xi32>)
                          outs(%operand: tensor<2xi32>) (%in: i32, %out: i32) {
    thlo.yield %in : i32
  }
  return %scatter : tensor<2xi32>
}

// CHECK-LABEL: @update_last_element
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 0]

func.func @bufferized() -> memref<10xi32> {
  %operand = arith.constant dense<0> : memref<10xi32>
  %indices = arith.constant dense<[[1], [8], [-1]]> : memref<3x1xindex>
  %updates = arith.constant dense<[[4, 5, 6], [6, 7, 8], [8, 9, 10]]> : memref<3x3xi32>
  thlo.scatter ins(%indices: memref<3x1xindex>, %updates: memref<3x3xi32>)
               outs(%operand: memref<10xi32>) (%in: i32, %out: i32) {
    %add = arith.addi %in, %out : i32
    thlo.yield %add : i32
  }
  return %operand : memref<10xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 4, 5, 6, 0, 0, 0, 0, 0, 0]
