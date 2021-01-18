// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s

// CHECK: error: operand result uses variant type. Currently, the variant type is not natively supported in TFLite. Please consider not using the variant type: 'tensor<!tf.variant<tensor<2xi32>>>'
func @main() -> tensor<!tf.variant<tensor<2xi32>>> {
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<!tf.variant>} : () -> tensor<!tf.variant<tensor<2xi32>>>
  return %0 : tensor<!tf.variant<tensor<2xi32>>>
}
