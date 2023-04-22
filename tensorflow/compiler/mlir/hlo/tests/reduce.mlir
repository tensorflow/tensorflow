// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

// -----

// CHECK-LABEL: func @noop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32>)
// CHECK: return %[[ARG0]]
func @noop(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[]> : tensor<0xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @and_fold
func @and_fold() -> (tensor<i1>, tensor<i1>) {
  %0 = mhlo.constant dense<true> : tensor<2xi1>
  %2 = mhlo.constant dense<true> : tensor<i1>
  %3 = mhlo.constant dense<false> : tensor<i1>
  %4 = "mhlo.reduce"(%0, %2) ( {
  ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
    %11 = mhlo.and %arg2, %arg3 : tensor<i1>
    "mhlo.return"(%11) : (tensor<i1>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi1>, tensor<i1>) -> tensor<i1>

  %5 = "mhlo.reduce"(%0, %3) ( {
  ^bb0(%arg4: tensor<i1>, %arg5: tensor<i1>):
    %12 = mhlo.and %arg4, %arg5 : tensor<i1>
    "mhlo.return"(%12) : (tensor<i1>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
  return %4, %5 : tensor<i1>, tensor<i1>

 // CHECK-DAG: %[[CST:.*]] = mhlo.constant dense<true> : tensor<i1>
 // CHECK-DAG: %[[CST1:.*]] = mhlo.constant dense<false> : tensor<i1>
 // CHECK: return %[[CST]], %[[CST1]] : tensor<i1>, tensor<i1>
}

// -----

// CHECK-LABEL: func @or_fold
func @or_fold() -> (tensor<i1>, tensor<i1>) {
  %0 = mhlo.constant dense<false> : tensor<2xi1>
  %2 = mhlo.constant dense<false> : tensor<i1>
  %3 = mhlo.constant dense<true> : tensor<i1>
  %4 = "mhlo.reduce"(%0, %2) ( {
  ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
    %11 = mhlo.or %arg2, %arg3 : tensor<i1>
    "mhlo.return"(%11) : (tensor<i1>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi1>, tensor<i1>) -> tensor<i1>

  %5 = "mhlo.reduce"(%0, %3) ( {
  ^bb0(%arg4: tensor<i1>, %arg5: tensor<i1>):
    %12 = mhlo.or %arg4, %arg5 : tensor<i1>
    "mhlo.return"(%12) : (tensor<i1>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
  return %4, %5 : tensor<i1>, tensor<i1>

 // CHECK-DAG: %[[CST:.*]] = mhlo.constant dense<false> : tensor<i1>
 // CHECK-DAG: %[[CST1:.*]] = mhlo.constant dense<true> : tensor<i1>
 // CHECK: return %[[CST]], %[[CST1]] : tensor<i1>, tensor<i1>
}
