// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Confirm float constants and operators survive a roundtrip

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  // CHECK:      [[CONST:%.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK-NEXT: [[SQDIFF:%.*]] = tfl.squared_difference %arg0, [[CONST]] : tensor<4xf32>
  // CHECK-NEXT: %{{.*}} = tfl.mul %arg0, [[SQDIFF]] {fused_activation_function = "NONE"} : tensor<4xf32>
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  // Confirm that attributes that cannot be stored in the flatbuffer options
  // for a given operator are dropped silently.
  %1 = "tfl.squared_difference"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  %2 = "tfl.mul"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  %3 = "tfl.div"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  %4 = "tfl.exp"(%3) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  %5 = "tfl.neg"(%4) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  return %5 : tensor<4xf32>
}
