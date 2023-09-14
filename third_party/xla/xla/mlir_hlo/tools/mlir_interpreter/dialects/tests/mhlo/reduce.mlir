// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @reduce() -> tensor<3xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %init = mhlo.constant dense<1> : tensor<i32>
  %reduce = mhlo.reduce(%cst init: %init) across dimensions = [0]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
    reducer(%arg0: tensor<i32>, %arg1: tensor<i32>)  {
      %0 = mhlo.add %arg0, %arg1 : tensor<i32>
      mhlo.return %0 : tensor<i32>
    }
  return %reduce : tensor<3xi32>
}

// CHECK-LABEL: @reduce
// CHECK-NEXT: Results
// CHECK-NEXT: [6, 8, 10]
