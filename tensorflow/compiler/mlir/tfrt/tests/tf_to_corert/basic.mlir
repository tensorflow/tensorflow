// RUN: tf-opt -tf-to-corert %s | FileCheck %s

// CHECK-NOT: tf_saved_model.semantics
module attributes {tf_saved_model.semantics} {

// CHECK-NOT: "tf_saved_model.global_tensor"
"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<1x3xf32>, value = dense<[[1.67482901, -0.529208779, -0.803792417]]> : tensor<1x3xf32>} : () -> ()
"tf_saved_model.global_tensor"() {is_mutable, sym_name = "z", type = tensor<3xf32>, value = dense<[1.67482901, -0.529208779, -0.803792417]> : tensor<3xf32>} : () -> ()

// CHECK-LABEL: func @basic
// CHECK-SAME: ([[arg0:%.*]]: !corert.tensorhandle, [[arg1:%.*]]: !corert.tensorhandle,
// CHECK-SAME: [[arg2:%.*]]: !corert.tensorhandle) -> !corert.tensorhandle {
func @func_basic(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y},
    %arg2: tensor<!tf.resource<tensor<3xf32>>> {tf_saved_model.bound_input = @z})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["basic"]} {
  // CHECK-NEXT: [[cpu_device:%.*]] = corert.get_device "cpu"
  // CHECK-NEXT: [[r0:%.*]] = corert.executeop([[cpu_device]]) "tf.MatMul"([[arg0]], [[arg1]])
  // CHECK-NEXT: [[r1:%.*]] = corert.executeop([[cpu_device]]) "tf.BiasAdd"([[r0]], [[arg2]])
  // CHECK-NEXT: [[r2:%.*]] = corert.executeop([[cpu_device]]) "tf.Tanh"([[r1]])
  // CHECK-NEXT: hex.return [[r2]] : !corert.tensorhandle

  %0 = "tf.ReadVariableOp"(%arg2) {_output_shapes = ["tfshape$dim { size: 3 }"], device = "cpu", dtype = f32} : (tensor<!tf.resource<tensor<3xf32>>>) -> tensor<3xf32>
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "cpu", dtype = f32} : (tensor<!tf.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %2 = "tf.MatMul"(%arg0, %1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  %3 = "tf.BiasAdd"(%2, %0) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], data_format = "NHWC", device = "cpu"} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
  %4 = "tf.Tanh"(%3) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %5 = "tf.Identity"(%4) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %5 : tensor<3x3xf32>
}

}
