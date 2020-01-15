// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() -> (tensor<1x2xf16>, tensor<2xf16>) {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<1.0> : tensor<1x2xf16>} : () -> (tensor<1x2xf16>, !_tf.control) loc("const1")
  %1:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> (tensor<2xf16>, !_tf.control) loc("const2")
	%2:2 = "_tf.Const"() {device = "", dtype = bf16, value = dense<[4.900000e+01, 8.200000e+02]> : tensor<2xbf16>} : () -> (tensor<bf16>, !_tf.control) loc("const3")
	%3:2 = "_tf.Const"() {device = "", dtype = bf16, value = dense<0.000000e+00> : tensor<bf16>} : () -> (tensor<bf16>, !_tf.control) loc("const4")
  return %0#0, %1#0 : tensor<1x2xf16>, tensor<2xf16>
}

// CHECK: node {
// CHECK-NEXT: name: "const1"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_HALF
// CHECK: half_val: 15360
// CHECK: name: "const2"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_HALF
// CHECK: half_val: 15360
// CHECK: half_val: 16384
// CHECK: name: "const3"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_BFLOAT16
// CHECK: half_val: 16964
// CHECK: half_val: 17485
// CHECK: name: "const4"
// CHECK-NEXT: op: "Const"
// CHECK: dtype: DT_BFLOAT16
// CHECK: half_val: 0
