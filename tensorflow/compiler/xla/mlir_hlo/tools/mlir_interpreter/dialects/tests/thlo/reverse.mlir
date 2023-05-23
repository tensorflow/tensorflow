// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @reverse() -> tensor<3x3xi32> {
  %tensor = arith.constant dense<[[4, 5, 6], [6, 7, 8], [8, 9, 10]]> : tensor<3x3xi32>
  %init = tensor.empty() : tensor<3x3xi32>
  %reverse = thlo.reverse ins(%tensor: tensor<3x3xi32>)
                          outs(%init: tensor<3x3xi32>)
                          reverse_dimensions = [1]
  return %reverse : tensor<3x3xi32>
}

// CHECK-LABEL: @reverse
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[6, 5, 4], [8, 7, 6], [10, 9, 8]]
