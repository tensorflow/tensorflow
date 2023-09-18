// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

//test TF ops wrapped in stablehlo custom_call

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

//CHECK:func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> attributes {tf.entry_function = {inputs = "arg0", outputs = "arg0"}} {
//CHECK: return %arg0 : tensor<4xi8>
//CHECK:}

func.func @custom_tf_op(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
  %0 = stablehlo.custom_call @tf.ResizeBilinear(%arg0, %arg1) {align_corners = true, device = "", half_pixel_centers = false} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
  return %0 : tensor<1x1600x1x1xf32>
}

//CHECK:func.func private @custom_tf_op(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
//CHECK-NEXT: %0 = stablehlo.custom_call @tf.ResizeBilinear(%arg0, %arg1) {align_corners = true, device = "", half_pixel_centers = false} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
//CHECK-NEXT: return %0 : tensor<1x1600x1x1xf32>
//CHECK-NEXT:}

func.func @custom_op_with_backend(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
  %0 = stablehlo.custom_call @custom_backend(%arg0, %arg1) {backend_config = "test", has_side_effect = true} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
  return %0 : tensor<1x1600x1x1xf32>
}

//CHECK:func.func private @custom_op_with_backend(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
//CHECK-NEXT: %0 = stablehlo.custom_call @custom_backend(%arg0, %arg1) {backend_config = "test", has_side_effect = true} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
//CHECK-NEXT: return %0 : tensor<1x1600x1x1xf32>
//CHECK-NEXT:}