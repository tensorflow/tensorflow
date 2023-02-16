// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @reduce() -> tensor<2xi32> {
  %v = arith.constant dense<[[1,2,3,4], [5,6,7,8]]> : tensor<2x4xi32>
  %init = arith.constant dense<[9, 10]> : tensor<2xi32>
  %ret = linalg.reduce ins(%v : tensor<2x4xi32>)
                       outs(%init: tensor<2xi32>)
                       dimensions = [1]
                       (%in: i32, %out: i32) {
                         %sum = arith.addi %in, %out : i32
                         linalg.yield %sum: i32
                       }
  func.return %ret : tensor<2xi32>
}

// CHECK-LABEL: @reduce
// CHECK-NEXT: Results
// CHECK-NEXT: [19, 36]

func.func @bufferized() -> memref<2xi32> {
  %v = arith.constant dense<[[1,2,3,4], [5,6,7,8]]> : tensor<2x4xi32>
  %init = arith.constant dense<[9, 10]> : memref<2xi32>
  linalg.reduce ins(%v : tensor<2x4xi32>)
                outs(%init: memref<2xi32>)
                dimensions = [1]
                (%in: i32, %out: i32) {
                  %sum = arith.addi %in, %out : i32
                  linalg.yield %sum: i32
                }
  func.return %init : memref<2xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT: [19, 36]

