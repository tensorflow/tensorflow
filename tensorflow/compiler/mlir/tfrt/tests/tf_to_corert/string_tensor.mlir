// RUN: tf-opt -tf-to-corert %s | FileCheck %s

// CHECK-LABEL: func @string_tensor
func @string_tensor() -> (tensor<0x!tf.string>, tensor<7x!tf.string>) {
  // CHECK: {shape = [0], value = []}
  %0 = "tf.Const"() {value = dense<[]> : tensor<0x!tf.string>} : () -> tensor<0x!tf.string>
  // CHECK: {shape = [7], value = ["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]}
  %1 = "tf.Const"() {value = dense<["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]> : tensor<7x!tf.string>} : () -> tensor<7x!tf.string>
  return %0, %1 : tensor<0x!tf.string>, tensor<7x!tf.string>
}
