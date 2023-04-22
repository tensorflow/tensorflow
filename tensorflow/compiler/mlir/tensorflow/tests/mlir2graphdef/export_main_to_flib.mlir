// RUN: tf-mlir-translate -mlir-to-graphdef -tf-export-entry-func-to-flib  %s -o - 2>&1 | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 458 : i32}} {
  func @main() attributes {tf.entry_function = {inputs = "", outputs = ""}} {
    tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "TPU:0", name = "const", dtype = "tfdtype$DT_INT32", value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
      tf_executor.fetch
    }
    return
  }
}

// CHECK-NOT: node

// CHECK: library
// CHECK-NEXT: function
// CHECK-NEXT: signature
// CHECK-NEXT: name: "main"
// CHECK: node_def
// CHECK: op: "Const"
