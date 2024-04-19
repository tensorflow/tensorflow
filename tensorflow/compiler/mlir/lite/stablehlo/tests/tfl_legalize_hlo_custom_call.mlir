// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: mhlo_custom_call_test__legalize_string_backend_config
func.func @mhlo_custom_call_test__legalize_string_backend_config(%arg0: tensor<1x4xf32>) -> tensor<1x8xf32> {
  %0 = mhlo.custom_call @custom_call.my_custom_op(%arg0) {
    api_version = 1 : i32,
    backend_config = "this_is_a_test_string"
  } : (tensor<1x4xf32>) -> (tensor<1x8xf32>)
  func.return %0 : tensor<1x8xf32>

  //       CHECK: %0 = "tfl.custom"(%arg0) {
  //  CHECK-SAME:   custom_code = "custom_call.my_custom_op",
  //  CHECK-SAME:   custom_option = #tfl<const_bytes : "0x746869735F69735F615F746573745F737472696E67">
  //  CHECK-SAME: } : (tensor<1x4xf32>) -> tensor<1x8xf32>
}

// CHECK-LABEL: mhlo_custom_call_test__dont_legalize_dict_backend_config
func.func @mhlo_custom_call_test__dont_legalize_dict_backend_config(%arg0: tensor<1x4xf32>) -> tensor<1x8xf32> {
  %0 = mhlo.custom_call @custom_call.my_custom_op(%arg0) {
    api_version = 4 : i32,
    backend_config = {foo = "bar"}
  } : (tensor<1x4xf32>) -> (tensor<1x8xf32>)
  func.return %0 : tensor<1x8xf32>

  //       CHECK: %0 = mhlo.custom_call @custom_call.my_custom_op(%arg0) {
  //  CHECK-SAME:   api_version = 4 : i32,
  //  CHECK-SAME:   backend_config = {foo = "bar"}
  //  CHECK-SAME: } : (tensor<1x4xf32>) -> tensor<1x8xf32>
}

// CHECK-LABEL: mhlo_custom_call_test__api_version_4
func.func @mhlo_custom_call_test__api_version_4(%arg0: tensor<1x4xf32>) -> tensor<1x8xf32> {
  %0 = mhlo.custom_call @custom_call.my_custom_op(%arg0) {
    api_version = 4 : i32
  } : (tensor<1x4xf32>) -> (tensor<1x8xf32>)
  func.return %0 : tensor<1x8xf32>

  //       CHECK: %0 = "tfl.custom"(%arg0) {
  //  CHECK-SAME:   custom_code = "custom_call.my_custom_op",
  //  CHECK-SAME:   custom_option = #tfl<const_bytes : "0x">
  //  CHECK-SAME: } : (tensor<1x4xf32>) -> tensor<1x8xf32>
}

// CHECK-LABEL: mhlo_custom_call_does_not_legalize_tf_function
func.func @mhlo_custom_call_does_not_legalize_tf_function(%arg0: tensor<1x4xf32>) -> tensor<1x8xf32> {
  %0 = mhlo.custom_call @tf.ResizeBilinear(%arg0) {
    backend_config = "this_is_a_test_string"
  } : (tensor<1x4xf32>) -> (tensor<1x8xf32>)
  func.return %0 : tensor<1x8xf32>

  //       CHECK: %0 = mhlo.custom_call @tf.ResizeBilinear(%arg0) {
  //  CHECK-SAME:   backend_config = "this_is_a_test_string"
  //  CHECK-SAME: } : (tensor<1x4xf32>) -> tensor<1x8xf32>
}
