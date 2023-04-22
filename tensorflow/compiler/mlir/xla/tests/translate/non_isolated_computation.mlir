// RUN: not tf-mlir-translate -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

func @main(%arg0: tensor<i64>) -> tensor<i64> {
  %c0 = mhlo.constant dense<1> : tensor<i64>
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    // CHECK: requires all operands to be defined in the parent region for export
    %1 = "mhlo.compare"(%c0, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %2 = mhlo.add %arg1, %arg1 : tensor<i64>
    "mhlo.return"(%2) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}
