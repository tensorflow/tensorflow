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

func.func @variadic() -> (tensor<2xi32>, tensor<2xf32>) {
  %v = arith.constant dense<[[1,2,3,4], [5,6,7,8]]> : tensor<2x4xi32>
  %w = arith.constant dense<[[1.0,2.0,3.0,4.0], [5.0,6.0,7.0,8.0]]> : tensor<2x4xf32>
  %init = arith.constant dense<[9, 10]> : tensor<2xi32>
  %init2 = arith.constant dense<[9.0, 10.0]> : tensor<2xf32>
  %ret, %retf = linalg.reduce ins(%v, %w : tensor<2x4xi32>, tensor<2x4xf32>)
                              outs(%init, %init2: tensor<2xi32>, tensor<2xf32>)
                              dimensions = [1]
    (%in: i32, %inf: f32, %out: i32, %outf: f32) {
      %sum = arith.addi %in, %out : i32
      %sumf = arith.addf %inf, %outf : f32
      linalg.yield %sum, %sumf: i32, f32
    }
  func.return %ret, %retf : tensor<2xi32>, tensor<2xf32>
}

// CHECK-LABEL: @variadic
// CHECK-NEXT: Results
// CHECK-NEXT: [19, 36]
// CHECK-NEXT: [1.900000e+01, 3.600000e+01]

func.func @bufferized() -> memref<2xi32> {
  %v = arith.constant dense<[[1,2,3,4], [5,6,7,8]]> : memref<2x4xi32>
  %init = arith.constant dense<[9, 10]> : memref<2xi32>
  linalg.reduce ins(%v : memref<2x4xi32>)
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

