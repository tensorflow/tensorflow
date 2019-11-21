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

// expected-error @+1 {{encountered unexpected token}}
func @invalid_type(!tf<"variant>">) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @invalid_type(!tf.variant<>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func @invalid_type(!tf.variant<tensor<??xf32>>) -> ()

// -----

// expected-error @+1 {{invalid kind of type specified}}
func @invalid_type(!tf.variant<vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{invalid VariantType subtype: 'tensor<vector<2xf32>>'}}
func @invalid_type(!tf.variant<tensor<vector<2xf32>>>) -> ()

// -----

// CHECK: !tf.resource
func @resource_without_type(!tf.resource) -> ()

// CHECK: !tf.resource<tensor<?xf32>>
func @resource_with_type(!tf.resource<tensor<?xf32>>) -> ()

// CHECK: !tf.resource<tensor<3xf32>, tensor<2xi32>>
func @resource_with_multiple_types(!tf.resource<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf.resource<tensor<?xf32>>>
func @resource_element_type(tensor<*x!tf.resource<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf.resource<tensor<?x!tf.resource<tensor<?xf32>>>>>
func @nested_resource(tensor<!tf.resource<tensor<?x!tf.resource<tensor<?xf32>>>>>) -> ()

// CHECK: !tf.resourceref
func @resourceref(!tf.resourceref) -> ()

// -----

// expected-error @+1 {{encountered unexpected token}}
func @invalid_type(!tf<"resource>">) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @invalid_type(!tf.resource<>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func @invalid_type(!tf.resource<tensor<??xf32>>) -> ()

// -----

// expected-error @+1 {{invalid kind of type specified}}
func @invalid_type(!tf.resource<vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{invalid ResourceType subtype: 'tensor<vector<2xf32>>'}}
func @invalid_type(!tf.resource<tensor<vector<2xf32>>>) -> ()

// -----
