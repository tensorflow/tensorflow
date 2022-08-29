// RUN: stablehlo-opt %s | FileCheck %s
// RUN: stablehlo-opt %s | stablehlo-opt | FileCheck %s

// CHECK-LABEL: func @while
func.func @while() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
// CHECK-DAG: %[[CST_M1:.*]] = arith.constant dense<-1>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<1>
// CHECK-DAG: %[[CST_1000:.*]] = arith.constant dense<1000>
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %cst_1 = arith.constant dense<0> : tensor<i32>
  %cst_2 = arith.constant dense<1000> : tensor<i32>
// CHECK: %[[WHILE:.*]]:3 = stablehlo.while
// CHECK-SAME: (%[[ITER_ARG:.*]] = %[[CST_0]],
// CHECK-SAME: %[[ITER_ARG1:.*]] = %[[CST_M1]],
// CHECK-SAME: %[[ITER_ARG2:.*]] = %[[CST_1000]])
// CHECK-SAME: tensor<i32>, tensor<i32>, tensor<i32>

  %0:3 = stablehlo.while(%iterArg = %cst_1, %iterArg_3 = %cst, %iterArg_4 = %cst_2) : tensor<i32>, tensor<i32>, tensor<i32>
// CHECK-NEXT: cond {
   cond  {
// CHECK-NEXT: stablehlo.compare
// CHECK-SAME: %[[ITER_ARG]], %[[ITER_ARG2]]
    %1 = "stablehlo.compare"(%iterArg, %iterArg_4) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%1) : (tensor<i1>) -> ()
  } do  {
// CHECK: stablehlo.add
// CHECK-SAME: %[[ITER_ARG]], %[[CST_1]]
    %1 = stablehlo.add %iterArg, %cst_0 : tensor<i32>
    "stablehlo.return"(%1, %iterArg_3, %iterArg_4) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }
  func.return %0#0, %0#2, %0#2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @while_no_arg
func.func @while_no_arg() {
// CHECK:  stablehlo.while()
  stablehlo.while()
  cond {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    "stablehlo.return"(%0) : (tensor<i1>) -> ()
  } do {
   "stablehlo.return"() : () -> ()
  }
  func.return
}
