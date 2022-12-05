// RUN: tf-mhlo-tfl-opt %s -tf-mhlo-tfl | FileCheck %s

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
// CHECK-DAG: %cst = arith.constant dense<1> : tensor<1xi32>
// CHECK-DAG: %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x1x2xf32>
// CHECK: %0 = "tf.InplaceUpdate"(%arg0, %cst, %cst_0) {device = ""} : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
// CHECK: return %0 : tensor<2x1x2xf32>
