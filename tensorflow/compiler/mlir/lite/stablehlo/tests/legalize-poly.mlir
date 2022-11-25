// RUN: tf-mhlo-tfl-opt %s -tf-poly | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {

func.func @tfInplaceUpdate(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
  %1 = arith.constant dense<1> : tensor<1xi32>
  %2 = arith.constant dense<2.0> : tensor<1x1x2xf32>
  %3 = "tf.InplaceUpdate"(%arg0, %1, %2) {device = ""}
    : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
  func.return %3 : tensor<2x1x2xf32>
}

}

// CHECK-LABEL: @tfInplaceUpdate
// CHECK-NEXT:  %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK-NEXT:  %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x1x2xf32>
// CHECK-NEXT:  %0 = "tfl.poly_call"(%arg0, %cst, %cst_0) ({
// CHECK-NEXT:   ^bb0(%arg1: tensor<2x1x2xf32>, %arg2: tensor<1xi32>, %arg3: tensor<1x1x2xf32>):
// CHECK-NEXT:    %1 = "tf.InplaceUpdate"(%arg1, %arg2, %arg3) {device = ""} : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
// CHECK-NEXT:    "tfl.yield"(%1) : (tensor<2x1x2xf32>) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:   ^bb0(%arg1: tensor<2x1x2xf32>, %arg2: tensor<1xi32>, %arg3: tensor<1x1x2xf32>):
// CHECK-NEXT:    %1 = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
// CHECK-NEXT:    %2 = mhlo.reshape %1 : (tensor<1xi32>) -> tensor<i32>
// CHECK-NEXT:    %3 = mhlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %4 = "mhlo.slice"(%arg3) {limit_indices = dense<[1, 1, 2]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %5 = mhlo.dynamic_update_slice %arg1, %4, %2, %3, %3 : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
// CHECK-NEXT:    "tfl.yield"(%5) : (tensor<2x1x2xf32>) -> ()
// CHECK-NEXT:  }) {device = ""} : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
// CHECK-NEXT:  return %0 : tensor<2x1x2xf32>
