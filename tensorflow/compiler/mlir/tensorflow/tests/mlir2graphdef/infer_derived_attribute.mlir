// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  // The operation does not have any attributes, but TensorFlow OpDef expects
  // a `dtype` to be added on the NodeDef. We verify that we correctly use the
  // DerivedAttr to populate the NodeDef.
  // CHECK:      key: "dtype"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   type: DT_FLOAT
  // CHECK:   float_val: 2
  // CHECK:      key: "dtype"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   type: DT_FLOAT
  // CHECK:   float_val: 3
  // CHECK:      key: "dtype"
  // CHECK-NEXT: value {
  // CHECK-NEXT:   type: DT_DOUBLE
  // CHECK:   double_val: 4
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1:2 = tf_executor.island(%0#1) wraps "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2:2 =  tf_executor.island(%1#1) wraps "tf.Const"() {value = dense<4.000000e+00> : tensor<f64>} : () -> tensor<f64>
    tf_executor.fetch
  }
  func.return
}
