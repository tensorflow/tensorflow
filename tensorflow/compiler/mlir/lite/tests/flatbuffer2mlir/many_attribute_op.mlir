// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

// Confirm a wide array of attribute survives the round-trip
func @main(tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16xf32>):
  // CHECK: "tfl.average_pool_2d"(%{{.*}}) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> loc("avgpool")
  return %0 : tensor<1x1x1x16xf32>
}
