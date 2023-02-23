// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @while() -> (tensor<i32>, tensor<i32>) {
  %c0 = mhlo.constant dense<0> : tensor<i32>
  %c1 = mhlo.constant dense<1> : tensor<i32>
  %c10 = mhlo.constant dense<10> : tensor<i32>
  %3:2 = "mhlo.while"(%c0, %c1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.compare"(%arg0, %c10) {
        comparison_direction = #mhlo<comparison_direction LT>
      } : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%4) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = mhlo.add %arg0, %c1 : tensor<i32>
      %6 = mhlo.add %arg1, %arg1 : tensor<i32>
      "mhlo.return"(%5, %6) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %3#0, %3#1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: @while
// CHECK-NEXT: Results
// CHECK-NEXT: TensorOrMemref<i32>: 10
// CHECK-NEXT: TensorOrMemref<i32>: 1024
