// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() -> (tensor<1x2xf16>, tensor<2xf16>) {
  %0:2 = "_tf.Const"() {device = "", name = "foo", dtype = "tfdtype$DT_HALF", value = dense<1.0> : tensor<1x2xf16>} : () -> (tensor<1x2xf16>, !_tf.control)
  %1:2 = "_tf.Const"() {device = "", name = "bar", dtype = "tfdtype$DT_HALF", value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> (tensor<2xf16>, !_tf.control)
  return %0#0, %1#0 : tensor<1x2xf16>, tensor<2xf16>

// CHECK: node {
// CHECK-NEXT: name: "foo"
// CHECK-NEXT: op: "Const"
// CHECK: half_val: 15360
// CHECK: name: "bar"
// CHECK-NEXT: op: "Const"
// CHECK: half_val: 15360
// CHECK: half_val: 16384
}