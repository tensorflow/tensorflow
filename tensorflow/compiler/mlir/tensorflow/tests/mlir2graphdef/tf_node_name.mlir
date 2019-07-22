// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// CHECK:  name: "foo"

func @main() {
    %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "foo", value = dense<1> : tensor<i32>} : () -> (tensor<i32>, !_tf.control)
    return
}
