// RUN: tf-tfrt-opt -split-input-file -tf-mlrt-rewrite-ifrt-load-variable %s | FileCheck %s

// Variable is used by both CPU and TPU
//
// CHECK-LABEL: func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    [[HANDLE:%.*]] = "tf.VarHandleOp"()
// CHECK-NEXT:    [[ARRAYKEY:%.*]], [[FURTURE:%.*]] = "tf_mlrt.tf_ifrt_load_variable"([[HANDLE]])
// CHECK-SAME:       {device_sharding_config_proto_text = "sharding { }", name = "__y"} : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> (tensor<!tf_type.string>, !mlrt.future)
// CHECK-NEXT:    [[TENSOR:%.*]] = "tf_mlrt.tf_await"([[FURTURE]]) : (!mlrt.future) -> tensor<3x1xf32>
// CHECK-NEXT:    "tf.MatMul"(%arg0, [[TENSOR]]) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    "tf.IfrtCall"(%arg0, [[ARRAYKEY]]) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [1 : i32]}> {__tpu_compile_metadata_text = "retvals { sharding { } }"} : (tensor<1x3xf32>, tensor<!tf_type.string>) -> tensor<1x1xf32>
// CHECK-NEXT:    return
//
 func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %array_key, %tensor = "tf.IfrtLoadVariable"(%0) <{device_sharding_config_proto_text = "sharding { }", name = "__y"}> : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> (tensor<!tf_type.string>, tensor<3x1xf32>)
    %1 = "tf.MatMul"(%arg0, %tensor) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %2 = "tf.IfrtCall"(%arg0, %array_key) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [1 : i32]}> {__tpu_compile_metadata_text = "retvals { sharding { } }"} : (tensor<1x3xf32>, tensor<!tf_type.string>) -> tensor<1x1xf32>
    return %2 : tensor<1x1xf32>
  }
