// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=8,16,16,64:64 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text-via-builder %s -tf-input-shapes=8,16,16,64:64 | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func @main(%arg0: tensor<8x16x16x64xbf16>, %arg1: tensor<64xf32>) -> (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>) {
    %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg1, %arg1, %arg1) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x16x16x64xbf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       -> (bf16[8,16,16,64], f32[64], f32[64], f32[64], f32[64], /*index=5*/f32[0])
