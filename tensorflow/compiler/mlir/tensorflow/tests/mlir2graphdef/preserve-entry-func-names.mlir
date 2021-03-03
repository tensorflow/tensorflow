// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "foo,bar", outputs = "Add"}} {
  %graph = tf_executor.graph {
    // This node would be renamed to bar1 [note: if imported from TF graphdef this would not be possible]
    %2:2 = tf_executor.island wraps "tf.Identity"(%arg1) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
    // The following node would be renamed to bar2
    %3:2 = tf_executor.island wraps "tf.Identity"(%2) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
    %4:2 = tf_executor.island wraps "tf.Add"(%arg0, %3) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
    tf_executor.fetch %4#0 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK: name: "foo"
// CHECK-NEXT: op: "_Arg"
// CHECK: name: "bar"
// CHECK-NEXT: op: "_Arg"
// CHECK: name: "[[BAR_ID_0:.*]]"
// CHECK-NEXT: op: "Identity"
// CHECK-NEXT: input: "bar"
// CHECK: name: "[[BAR_ID_1:.*]]"
// CHECK-NEXT: op: "Identity"
// CHECK-NEXT: input: "[[BAR_ID_0]]"
// CHECK: name: "Add"
// CHECK-NEXT: op: "_Retval"
// CHECK-NEXT: input: "Add1"
