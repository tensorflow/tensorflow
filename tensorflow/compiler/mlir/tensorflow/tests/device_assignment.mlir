// RUN: tf-opt -tf-simple-device-assignment='default-device=gpu' %s | FileCheck %s

// CHECK-LABEL: func @device_test
func.func @device_test(%arg0: tensor<3x1xf32>) -> (tensor<3x3xf32>) {

  // CHECK: device = "gpu"
  %0 = "tf.Const"() {value = dense<[[1.0, 2.0, 3.0]]> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  // CHECK: device = "gpu"
  %1 = "tf.MatMul"(%arg0, %0) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  // CHECK: device = "cpu"
  %2 = "tf.Relu"(%1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK: device = "gpu"
  %3 = "tf.Relu"(%2) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"]} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  func.return %3 : tensor<3x3xf32>
}
