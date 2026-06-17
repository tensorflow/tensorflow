// RUN: tf-opt %s -split-input-file -tf-verify-for-export -verify-diagnostics | FileCheck %s

module {
  func.func @failsNoIslands() {
    // expected-error @+1 {{functions must be of a single Graph with single op Islands: first op in function is not a tf_executor.graph}}
    func.return
  }
}

// -----

module {
  // CHECK-LABEL: func @passesSingleIslandOp
  func.func @passesSingleIslandOp() {
    // CHECK: _class = ["loc:@class"]
    tf_executor.graph {
      %c, %control0 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %a, %control1 = tf_executor.island wraps "tf.A"() {_class = ["loc:@class"]} : () -> (tensor<2xf32>)
      %s:2, %control2 = tf_executor.island wraps "tf.Split"(%c, %a) {num_split = 2 : i32} : (tensor<i32>, tensor<2xf32>) -> (tensor<1xf32>, tensor<1xf32>)
      tf_executor.fetch
    }
    func.return
  }
}