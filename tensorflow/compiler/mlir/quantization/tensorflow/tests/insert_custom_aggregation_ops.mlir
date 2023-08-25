// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops | FileCheck %s

module {
  func.func @add_custom_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @no_custom_ops_on_non_f32_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %add : tensor<*xi32>
  }

}

// CHECK: func @add_custom_ops
// CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  return [[res]] : tensor<*xf32>

// CHECK: func @no_custom_ops_on_non_f32_type
// CHECK-NEXT:  "tf.AddV2"
// CHECK-NEXT:  return
