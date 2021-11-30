// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @string_tensor
func @string_tensor() -> (tensor<0x!tf_type.string>, tensor<7x!tf_type.string>) {
  // CHECK: {shape = [0], value = []}
  %0 = "tf.Const"() {value = dense<[]> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // CHECK: {shape = [7], value = ["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]}
  %1 = "tf.Const"() {value = dense<["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]> : tensor<7x!tf_type.string>} : () -> tensor<7x!tf_type.string>
  return %0, %1 : tensor<0x!tf_type.string>, tensor<7x!tf_type.string>
}

// Convert tf.Const to corert.const_dense_tensor only on cpu device
// CHECK-LABEL: func @dense_tensor
func @dense_tensor() -> tensor<4xui64> {
  // CHECK: corert.const_dense_tensor dense<[1, 2, 3, 4]> : tensor<4xui64>
  %0 = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xui64>} : () -> tensor<4xui64>
  // CHECK: corert.const_dense_tensor  dense<1.000000e+00> : tensor<1xbf16>
  %1 = "tf.Const"() {device = "/device:CPU:0", value = dense<[1.0]> : tensor<1xbf16>} : () -> tensor<4xbf16>
  // CHECK: corert.executeop
  %2 = "tf.Const"() {device = "/device:GPU:0", value = dense<[1, 2, 3, 4]> : tensor<4xui64>} : () -> tensor<4xui64>
  return %0 : tensor<4xui64>
}

// CHECK-LABEL: func @tensor_proto
func @tensor_proto() -> tensor<!tf_type.quint8> {
  // tfrt_fallback_async.const_tensor_proto accepts a serialized tensor proto.
  // CHECK: tfrt_fallback_async.const_tensor_proto "\08\0C\12\00\22\01@"
  %0 = "tf.Const"() {value = opaque<"tf", "0x746674656E736F722464747970653A2044545F5155494E54382074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20224022"> : tensor<!tf_type.quint8>} : () -> tensor<!tf_type.quint8>
  return %0 : tensor<!tf_type.quint8>
}
