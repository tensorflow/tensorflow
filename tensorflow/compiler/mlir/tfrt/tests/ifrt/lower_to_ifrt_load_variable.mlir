// RUN: tf-tfrt-opt -split-input-file -lower-to-ifrt-load-variable %s | FileCheck %s

// -----
// Basic test: variables tensors for devices
//
// CHECK-LABEL: serving_default
// CHECK-NEXT:  "tf.VarHandleOp"()
// CHECK-NEXT:  "tf.ReadVariableOp"
// CHECK-NEXT:  "tf.IfrtLoadVariable"
// CHECK-SAME:       device_sharding_config_proto_text = "sharding { } device_ids: 0 device_ids: 1 ", name = "__y"
// CHECK-NEXT:   "tf.MatMul"
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> {__variable_array_name = "__y", __variable_sharding_config_text = "sharding { } device_ids: 0 device_ids: 1 ", __variable_used_by_device = true, __variable_used_by_host = true}: () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {__variable_array_name = "__y", __variable_sharding_config_text = "sharding { } device_ids: 0 device_ids: 1 ", __variable_used_by_device = true, __variable_used_by_host = true} : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %2 = "tf.MatMul"(%arg0, %1) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
}

// -----
// Basic test: variables tensors not used by devices are not loaded as array.
//
// CHECK-LABEL: serving_default
// CHECK-NOT:  "tf.IfrtLoadVariable"
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %2 = "tf.MatMul"(%arg0, %1) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
}



