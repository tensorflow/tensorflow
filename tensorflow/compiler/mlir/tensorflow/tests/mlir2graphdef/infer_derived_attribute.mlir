// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
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
  %0:2 = "_tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> (tensor<f32>, !_tf.control)
  %1:2 = "_tf.Const"(%0#1) {value = dense<3.000000e+00> : tensor<f32>} : (!_tf.control) -> (tensor<f32>, !_tf.control)
  %2:2 = "_tf.Const"(%1#1) {value = dense<4.000000e+00> : tensor<f64>} : (!_tf.control) -> (tensor<f64>, !_tf.control)
  return
}


