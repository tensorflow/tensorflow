// RUN: not flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s --dump-input-on-failure

// CHECK: error: 'tf.MyCustomOp' op is neither a custom op nor a flex op
// CHECK: error: failed while converting: 'main'
// CHECK: Ops that need custom implementation (enabled via setting the -emit-custom-ops flag):
// CHECK: tf.MyCustomOp {name = "MyCustomOp"}

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "tf.MyCustomOp"(%1, %0) {name = "MyCustomOp"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "tfl.exp"(%2) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
