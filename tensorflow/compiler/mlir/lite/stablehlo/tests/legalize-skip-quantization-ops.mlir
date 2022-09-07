// RUN: tf-mhlo-tfl-opt %s --tf-mhlo=skip-quantization-ops=true | FileCheck %s --check-prefix=CHECK-SKIP
// RUN: tf-mhlo-tfl-opt %s --tf-mhlo=skip-quantization-ops=false | FileCheck %s --check-prefix=CHECK-NOSKIP

func.func @fake_quant_with_min_max_vars(%arg0: tensor<1x1x28x48xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1x1x28x48xf32> {
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {device = "", narrow_range = true, num_bits = 8 : i64} : (tensor<1x1x28x48xf32>, tensor<f32>, tensor<f32>) -> tensor<1x1x28x48xf32>
  func.return %0 : tensor<1x1x28x48xf32>
  // CHECK-SKIP: tf.FakeQuantWithMinMaxVars
  // CHECK-NOSKIP-NOT: tf.
}
