// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s

// CHECK: function argument uses variant type. Currently, the variant type is not natively supported in TFLite. Please consider not using the variant type: 'tensor<!tf.variant<tensor<2xi32>>>'
func @main(%arg0 : tensor<!tf.variant<tensor<2xi32>>>) -> tensor<!tf.variant<tensor<2xi32>>> {
  return %arg0 : tensor<!tf.variant<tensor<2xi32>>>
}
