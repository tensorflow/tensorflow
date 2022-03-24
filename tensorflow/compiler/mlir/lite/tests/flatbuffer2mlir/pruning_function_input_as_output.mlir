// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate -output-arrays=mul,div,exp --experimental-prune-unreachable-nodes-unconditionally --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK: (%[[ARG:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {tf.entry_function = {inputs = "mul"}} {
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.squared_difference"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  // CHECK: %[[DIV:.*]] = tfl.div
  %2 = "tfl.div"(%1, %arg0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  // CHECK: %[[EXP:.*]] = "tfl.exp"
  %3 = "tfl.exp"(%2) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  // tfl.neg should be pruned
  // CHECK-NOT: "tfl.neg"
  %4 = "tfl.neg"(%3) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  // CHECK: return %[[ARG]], %[[DIV]], %[[EXP]]
  func.return %4 : tensor<4xf32>
}
