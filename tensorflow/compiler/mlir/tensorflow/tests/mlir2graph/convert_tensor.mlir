// RUN: tf-mlir-translate -mlir-to-graph %s -o - | FileCheck %s

func.func @main() -> (tensor<1x2xf16>, tensor<2xf16>) {
  %graph:2 = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<1.0> : tensor<1x2xf16>} : () -> tensor<1x2xf16> loc("const1")
// CHECK: node {
// CHECK-NEXT: name: "const1"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_HALF
// CHECK: half_val: 15360
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> tensor<2xf16> loc("const2")
// CHECK: name: "const2"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_HALF
// CHECK: half_val: 15360
// CHECK-NEXT: half_val: 16384
    %2:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = bf16, value = dense<[4.900000e+01, 8.200000e+02]> : tensor<2xbf16>} : () -> tensor<bf16> loc("const3")
// CHECK: name: "const3"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_BFLOAT16
// CHECK: half_val: 16964
// CHECK: half_val: 17485
    %3:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = bf16, value = dense<0.000000e+00> : tensor<bf16>} : () -> tensor<bf16> loc("const4")
// CHECK: name: "const4"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_BFLOAT16
// CHECK-NOT: half_val: 0
    tf_executor.fetch %0#0, %1#0 : tensor<1x2xf16>, tensor<2xf16>
  }
  func.return %graph#0, %graph#1 : tensor<1x2xf16>, tensor<2xf16>
}
