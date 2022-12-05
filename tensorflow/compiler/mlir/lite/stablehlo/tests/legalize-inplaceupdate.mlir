// RUN: tf-mhlo-tfl-opt %s -tf-mhlo | FileCheck %s

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
// CHECK-DAG: %{{.*}} = arith.constant dense<1> : tensor<1xi32>
// CHECK-DAG: %{{.*}} = arith.constant dense<2.000000e+00> : tensor<1x1x2xf32>
// CHECK-DAG: %[[CST0:.*]] = mhlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[CST1:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK-DAG: %[[CST2:.*]] = mhlo.constant dense<2.000000e+00> : tensor<1x1x2xf32>
// CHECK: %[[RES:.*]] = mhlo.dynamic_update_slice %arg0, %[[CST2]], %[[CST0]], %[[CST1]], %[[CST1]] : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
// CHECK: return %[[RES]] : tensor<2x1x2xf32>
