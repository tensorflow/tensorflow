// RUN: mlir-hlo-opt %s | FileCheck %s
// RUN: mlir-hlo-opt %s | mlir-hlo-opt | FileCheck %s


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
// CHECK: %[[WHILE:.*]]:3 = mhlo.while
// CHECK-SAME: (%[[ITER_ARG:.*]] = %[[CST_0]],
// CHECK-SAME: %[[ITER_ARG1:.*]] = %[[CST_M1]],
// CHECK-SAME: %[[ITER_ARG2:.*]] = %[[CST_1000]])
// CHECK-SAME: tensor<i32>, tensor<i32>, tensor<i32>

  %0:3 = mhlo.while(%iterArg = %cst_1, %iterArg_3 = %cst, %iterArg_4 = %cst_2) : tensor<i32>, tensor<i32>, tensor<i32>
// CHECK-NEXT: cond {
   cond  {
// CHECK-NEXT: mhlo.compare
// CHECK-SAME: %[[ITER_ARG]], %[[ITER_ARG2]]
    %1 = "mhlo.compare"(%iterArg, %iterArg_4) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  } do  {
// CHECK: mhlo.add
// CHECK-SAME: %[[ITER_ARG]], %[[CST_1]]
    %1 = mhlo.add %iterArg, %cst_0 : tensor<i32>
    "mhlo.return"(%1, %iterArg_3, %iterArg_4) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }
  func.return %0#0, %0#2, %0#2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @while_no_arg
func.func @while_no_arg() {
// CHECK:  mhlo.while()
  mhlo.while()
  cond {
    %0 = mhlo.constant dense<false> : tensor<i1>
    "mhlo.return"(%0) : (tensor<i1>) -> ()
  } do {
   "mhlo.return"() : () -> ()
  }
  func.return
}
