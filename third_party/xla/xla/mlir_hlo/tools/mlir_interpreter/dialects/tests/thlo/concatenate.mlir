// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @concatenate() -> tensor<5xi32> {
  %a = arith.constant dense<[1,2,3]> : tensor<3xi32>
  %b = arith.constant dense<[4,5]> : tensor<2xi32>
  %init = tensor.empty() : tensor<5xi32>
  %cat = thlo.concatenate
      ins(%a: tensor<3xi32>, %b: tensor<2xi32>)
      outs(%init: tensor<5xi32>)
      dimension = 0
  return %cat : tensor<5xi32>
}

// CHECK-LABEL: @concatenate
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 2, 3, 4, 5]
