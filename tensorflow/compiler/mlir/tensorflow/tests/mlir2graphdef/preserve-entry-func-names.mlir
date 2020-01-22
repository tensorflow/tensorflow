// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "foo,bar", outputs = "Add"}} {
  %0 = "tf.Placeholder.input"(%arg0) {device = "", dtype = "tfdtype$DT_INT32", shape = "tfshape$dim { size: 10 }"} : (tensor<10xi32>) -> tensor<10xi32>
  %1 = "tf.Placeholder.input"(%arg1) {device = "", dtype = "tfdtype$DT_INT32", shape = "tfshape$dim { size: 10 }"} : (tensor<10xi32>) -> tensor<10xi32>
  // This node would be renamed to bar1
  %2 = "tf.Identity"(%1) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
  // The following node would be renamed to bar2
  %3 = "tf.Identity"(%2) {device = "", dtype = "tfdtype$DT_INT32"} : (tensor<10xi32>) -> tensor<10xi32> loc ("bar")
  %4 = "tf.Add"(%0, %3) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
  return %4 : tensor<10xi32>
}

// CHECK: name: "bar1"
// CHECK-NEXT: op: "Identity"
// CHECK: name: "bar2"
// CHECK-NEXT: op: "Identity"
// CHECK: name: "Add"
// CHECK-NEXT: op: "Add"
// CHECK: name: "foo"
// CHECK-NEXT: op: "Placeholder"
// CHECK: name: "bar"
// CHECK-NEXT: op: "Placeholder"
