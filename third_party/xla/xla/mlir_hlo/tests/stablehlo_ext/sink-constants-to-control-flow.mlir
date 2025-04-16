// RUN: mlir-hlo-opt %s -stablehlo-ext-sink-constants-to-control-flow | FileCheck %s

// Tests that constants are not sunk to while loops, HLO lowering converts these
// to input parameters which results in faster execution.

// CHECK-LABEL: func @sink_const_to_while
func.func @sink_const_to_while(%arg0: tensor<i64>) -> tensor<i64> {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  // CHECK: stablehlo.while
  // CHECK-SAME: (%[[ITER_ARG:.*]] = %[[ARG1A:.+]]
  // CHECK: stablehlo.constant
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
    %1 = stablehlo.compare  LT, %c, %iterArg : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  // CHECK{LITERAL}: } do {
  // CHECK: stablehlo.constant
  } do {
    %1 = stablehlo.add %iterArg, %iterArg : tensor<i64>
    %2 = stablehlo.add %1, %c_0 : tensor<i64>
    %3 = stablehlo.add %2, %c_0 : tensor<i64>
    stablehlo.return %3 : tensor<i64>
  }
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @sink_const_to_conditional
func.func @sink_const_to_conditional(%arg0: tensor<i64>) -> tensor<i64> {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.compare  LT, %arg0, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: stablehlo.if
  %1 = "stablehlo.if"(%0) ({
    // CHECK-NEXT: stablehlo.constant dense<1> : tensor<i64>
    %2 = stablehlo.add %arg0, %c : tensor<i64>
    stablehlo.return %2 : tensor<i64>
  // CHECK{LITERAL}: }, {
  // CHECK-NEXT: stablehlo.constant dense<2> : tensor<i64>
  }, {
    %2 = stablehlo.add %arg0, %c_0 : tensor<i64>
    stablehlo.return %2 : tensor<i64>
  }) : (tensor<i1>) -> tensor<i64>
  return %1 : tensor<i64>
}

// CHECK-LABEL: func @sink_const_to_sort
func.func @sink_const_to_sort(%arg0: tensor<16xf32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.sort
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT: arith.constant dense<{{.*}}> : tensor<f32>
  %0 = "stablehlo.sort"(%arg0) <{is_stable = true}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.divide %arg1, %cst : tensor<f32>
    %2 = stablehlo.divide %arg2, %cst : tensor<f32>
    %3 = stablehlo.compare  GT, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %3 : tensor<i1>
  }) : (tensor<16xf32>) -> tensor<16xf32>
  return
}
