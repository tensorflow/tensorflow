// RUN: tf-tfrt-opt -tf-to-tfrt=func-use-fallback-tensor=true %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @device_test
func.func @device_test(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<1x3xf32> {tf_saved_model.index_path = [0]})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []}) {
  // CHECK: {{%.*}} = corert.get_op_handler %arg0 "/device:GPU:0"
  %2 = "tf.MatMul"(%arg0, %arg1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:GPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  func.return %2 : tensor<3x3xf32>
}
