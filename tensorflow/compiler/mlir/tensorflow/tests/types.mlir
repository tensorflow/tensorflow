// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: !tf.variant
func @variant_without_type(!tf.variant) -> ()

// CHECK: !tf.variant<tensor<?xf32>>
func @variant_with_type(!tf.variant<tensor<?xf32>>) -> ()

// CHECK: !tf.variant<tensor<3xf32>, tensor<2xi32>>
func @variant_with_multiple_types(!tf.variant<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf.variant<tensor<?xf32>>>
func @variant_element_type(tensor<*x!tf.variant<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf.variant<tensor<?x!tf.variant<tensor<?xf32>>>>>
func @nested_variant(tensor<!tf.variant<tensor<?x!tf.variant<tensor<?xf32>>>>>) -> ()

// CHECK: !tf.variantref
func @variantref(!tf.variantref) -> ()

// -----

// expected-error @+1 {{tf.variant delimiter <...> mismatch}}
func @invalid_type(!tf<"variant>">) -> ()

// -----

// expected-error @+1 {{invalid type: tf.variant<>}}
func @invalid_type(!tf.variant<>) -> ()

// -----

// expected-error @+1 {{invalid type: tensor<??xf32>}}
func @invalid_type(!tf.variant<tensor<??xf32>>) -> ()

// -----

// expected-error @+1 {{expected TensorType. Found: 'vector<3xf32>'}}
func @invalid_type(!tf.variant<vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{invalid VariantType subtype: 'tensor<vector<2xf32>>'}}
func @invalid_type(!tf.variant<tensor<vector<2xf32>>>) -> ()

// -----
