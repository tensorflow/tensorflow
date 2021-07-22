// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @device_test
func @device_test(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<1x3xf32> {tf_saved_model.index_path = [0]})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []}) {
  // CHECK: {{%.*}} = corert.get_op_handler %arg0 "/device:GPU:0"

  %2 = "tf.MatMul"(%arg0, %arg1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:GPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}

// CHECK-LABEL: func @legacy_device_name
func @legacy_device_name(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<1x3xf32> {tf_saved_model.index_path = [0]})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []}) {
  // CHECK: {{%.*}} = corert.get_op_handler %arg0 "/device:GPU:0"

  %2 = "tf.MatMul"(%arg0, %arg1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:GPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}
