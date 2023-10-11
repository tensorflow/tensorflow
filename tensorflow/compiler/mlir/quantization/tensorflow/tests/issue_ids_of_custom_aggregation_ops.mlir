// RUN: tf-quant-opt %s -quant-issues-ids-of-custom-aggregation-ops | FileCheck %s

func.func @issue_ids(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.CustomAggregator"(%arg1) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "tf.CustomAggregator"(%arg0) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "tf.AddV2"(%1, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.CustomAggregator"(%2) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}


// CHECK: func @issue_ids
// CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = "0"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = "1"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = "2"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  return [[res]] : tensor<*xf32>
