// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @pad() -> tensor<1x?x?xi32> {
  %it = arith.constant dense<[[[1, 2, 3], [2, 3, 4]]]> : tensor<1x2x3xi32>
  %offset = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %out = tensor.pad %it low[%c0, %offset, 0] high[0, %c0, %offset]  {
    ^bb0(%a: index, %b: index, %c: index):
      %c5 = arith.constant 5 : i32
      tensor.yield %c5 : i32
    } : tensor<1x2x3xi32> to tensor<1x?x?xi32>
  return %out : tensor<1x?x?xi32>
}

// CHECK-LABEL: @pad
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[5, 5, 5, 5, 5],
// CHECK-SAME{LITERAL}:   [5, 5, 5, 5, 5],
// CHECK-SAME{LITERAL}:   [1, 2, 3, 5, 5],
// CHECK-SAME{LITERAL}:   [2, 3, 4, 5, 5]]]

func.func @pad_args() -> tensor<3x3xindex> {
  %it = arith.constant dense<[[999]]> : tensor<1x1xindex>
  %out = tensor.pad %it low[1, 1] high[1, 1]  {
    ^bb0(%a: index, %b: index):
      %c10 = arith.constant 10 : index
      %mul = arith.muli %a, %c10 : index
      %ret = arith.addi %mul, %b : index
      tensor.yield %ret : index
    } : tensor<1x1xindex> to tensor<3x3xindex>
  return %out : tensor<3x3xindex>
}

// CHECK-LABEL: @pad_args
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 1, 2],
// CHECK-SAME{LITERAL}:  [10, 999, 12],
// CHECK-SAME{LITERAL}:  [20, 21, 22]]