// RUN: tf-tosa-opt %s --tosa-tflite-verify-fully-converted --split-input-file -verify-diagnostics


// CHECK-LABEL: func.func @main
func.func @main(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
  // CHECK: "tosa.add"
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// expected-error@below {{The following illegal operations still remain}}
func.func @main(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32> attributes {tf.entry_function = {inputs = "input", outputs = "output"}} {
  // expected-error@+1 {{'tfl.add' op : illegal op still exists}}
  %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
  // expected-error@+1 {{'tfl.sub' op : illegal op still exists}}
  %1 = tfl.sub %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
  return %1 : tensor<1x8x8x3xf32>
}
