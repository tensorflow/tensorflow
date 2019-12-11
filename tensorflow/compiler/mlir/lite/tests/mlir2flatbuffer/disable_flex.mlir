// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s 2>&1 | FileCheck %s; if [[ ${PIPESTATUS[0]} != 0  &&  ${PIPESTATUS[1]} == 0 ]]; then exit 0; else exit 1; fi

// CHECK: error: 'tf.Div' op is neither a custom op nor a flex op
// CHECK: error: failed while converting: 'main'
// CHECK: Ops that can be supported by the flex runtime (enabled via setting the -emit-select-tf-ops flag): Div.

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  %0 = "tfl.pseudo_const" () {name = "Const", value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE", name = "mul"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // tf.div is the result of conversion to a Flex TF op
  %2 = "tf.Div"(%1, %0) {name = "div"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "tfl.exp"(%2) {name = "exp"} : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
