// RUN: tf-mlir-translate -mlir-to-graphdef %s -tf-graph-as-function -o - | FileCheck %s

// Verify arg attributes are exported as device assignment for arg nodes.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 121 : i32}} {
  func @main(%arg0: tensor<*xf32> {tf.device = "/CPU:0"}, %arg1: tensor<2x4x6x8xi32>) -> (tensor<*xf32>, tensor<2x4x6x8xi32>)
  attributes  {tf.entry_function = {inputs = "args_0,args_1", outputs = "rets_0,rets_1"}} {
    %0:2 = tf_executor.graph {
      %1:3 = tf_executor.island wraps "tf.IdentityN"(%arg0, %arg1) {T = ["tfdtype$DT_FLOAT", "tfdtype$DT_INT32"], device = "", name = "identity"} : (tensor<*xf32>, tensor<2x4x6x8xi32>) -> (tensor<*xf32>, tensor<2x4x6x8xi32>)
      tf_executor.fetch %1#0, %1#1 : tensor<*xf32>, tensor<2x4x6x8xi32>
    }
    return %0#0, %0#1 : tensor<*xf32>, tensor<2x4x6x8xi32>
  }
}

// CHECK:      node {
// CHECK-NEXT:   name: "args_0"
// CHECK:        device: "/CPU:0"
// CHECK:        attr {
// CHECK:          key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 0
// CHECK_NEXT:     }
//
// CHECK:      node {
// CHECK-NEXT:   name: "args_1"
// CHECK-NOT:    device: "/CPU:0"
// CHECK:        attr {
// CHECK:          key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 1
// CHECK_NEXT:     }
