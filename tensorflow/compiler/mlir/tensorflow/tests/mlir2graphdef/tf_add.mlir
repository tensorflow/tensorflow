// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0,input1", outputs = "Add"}} {
  %graph = tf_executor.graph {
    %2:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
    tf_executor.fetch %2 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "input0"
// CHECK-NEXT:   op: "_Arg"
// CHECK:      node {
// CHECK-NEXT:   name: "input1"
// CHECK-NEXT:   op: "_Arg"
// CHECK:      node {
// CHECK-NEXT:   name: "Add1"
// CHECK-NEXT:   op: "Add"
// CHECK-NEXT:   input: "input0"
// CHECK-NEXT:   input: "input1"
// CHECK:      node {
// CHECK-NEXT:   name: "Add"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "Add1"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: library {
// CHECK-NEXT: }
