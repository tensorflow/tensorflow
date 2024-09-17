// RUN: tf-tfrt-opt -tf-mlrt-ifrt-set-tpu-host-allocator %s | FileCheck %s --dump-input=fail --dump-input-filter=all

// All arguments are non-variables

// CHECK-LABEL: func.func @serving_default
// CHECK-NEXT:    "tf.MatMul"
// CHECK-SAME:      tf_mlrt.custom_device = "tpu_host_device"
// CHECK-NEXT:    "tf.MatMul"
// CHECK-SAME:      tf_mlrt.custom_device = "tpu_host_device"
// CHECK-NEXT:    "tf.IfrtCall"
func.func @serving_default(%arg0: tensor<3x1xf32>,  %arg1: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
  %producer_0=  "tf.MatMul"(%arg1, %arg0) {transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %producer_1=  "tf.MatMul"(%arg1, %arg0) {transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %result = "tf.IfrtCall"(%producer_0, %producer_1) <{program_id = 6515870160938153680 : i64, variable_arg_indices = []}> : (tensor<1x1xf32>, tensor<1x1xf32>) -> (tensor<1x1xf32>)
  return %result: tensor<1x1xf32>
}

// Arguments to the IfrtCall are a mix of non-variables and variables

// CHECK-LABEL: func.func @serving_default1
// CHECK-NEXT:    "tf.VarHandleOp"
// CHECK-NOT:      tf_mlrt.custom_device
// CHECK-NEXT:    "tf.ReadVariableOp"
// CHECK-NOT:      tf_mlrt.custom_device
// CHECK-NEXT:    "tf.MatMul"
// CHECK-SAME:      tf_mlrt.custom_device = "tpu_host_device"
// CHECK-NEXT:    "tf.MatMul"
// CHECK-NOT:      tf_mlrt.custom_device
// CHECK-NEXT:    "tf.IfrtCall"
func.func @serving_default1(%arg0: tensor<3x1xf32>,  %arg1: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %2 = "tf.MatMul"(%arg1, %arg0) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %3 = "tf.MatMul"(%arg1, %arg0) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %result = "tf.IfrtCall"(%1, %2) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [0 : i32]}> : (tensor<3x1xf32>, tensor<1x1xf32>) -> (tensor<1x1xf32>)
    return %result: tensor<1x1xf32>
}



