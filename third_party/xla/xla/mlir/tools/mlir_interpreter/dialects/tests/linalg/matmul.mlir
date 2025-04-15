// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @m1x1() -> (tensor<1x1xi32>, tensor<1x1xi32>) {
  %a = arith.constant dense<1> : tensor<1x1xi32>
  %b = arith.constant dense<2> : tensor<1x1xi32>
  %c = arith.constant dense<3> : tensor<1x1xi32>
  %ret = linalg.matmul ins(%a, %b : tensor<1x1xi32>, tensor<1x1xi32>)
                       outs(%c : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %c, %ret : tensor<1x1xi32>, tensor<1x1xi32>
}

// CHECK-LABEL: @m1x1
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[3]]
// CHECK-NEXT{LITERAL}: [[5]]

func.func @m2x2() -> tensor<2x2xi32> {
  %a = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %b = arith.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi32>
  %c = tensor.empty() : tensor<2x2xi32>
  %ret = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
                       outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
  return %ret : tensor<2x2xi32>
}

// CHECK-LABEL: @m2x2
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[16, 19], [36, 43]]

func.func @m1x1_bufferized() -> memref<1x1xi32> {
  %a = arith.constant dense<1> : memref<1x1xi32>
  %b = arith.constant dense<2> : memref<1x1xi32>
  %c = arith.constant dense<3> : memref<1x1xi32>
  linalg.matmul ins(%a, %b : memref<1x1xi32>, memref<1x1xi32>)
                outs(%c : memref<1x1xi32>)
  return %c : memref<1x1xi32>
}

// CHECK-LABEL: @m1x1_bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[5]]
