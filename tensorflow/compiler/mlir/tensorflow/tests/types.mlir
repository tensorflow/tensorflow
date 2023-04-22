// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: !tf.variant
func private @variant_without_type(!tf.variant) -> ()

// CHECK: !tf.variant<tensor<?xf32>>
func private @variant_with_type(!tf.variant<tensor<?xf32>>) -> ()

// CHECK: !tf.variant<tensor<3xf32>, tensor<2xi32>>
func private @variant_with_multiple_types(!tf.variant<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf.variant<tensor<?xf32>>>
func private @variant_element_type(tensor<*x!tf.variant<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf.variant<tensor<?x!tf.variant<tensor<?xf32>>>>>
func private @nested_variant(tensor<!tf.variant<tensor<?x!tf.variant<tensor<?xf32>>>>>) -> ()

// CHECK: !tf.variantref
func private @variantref(!tf.variantref) -> ()

// -----

// expected-error @+1 {{unexpected token}}
func private @invalid_type(!tf<"variant>">) -> ()

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf.variant<>) -> ()

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf.variant<tensor<??xf32>>) -> ()

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf.variant<vector<3xf32>>) -> ()

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf.variant<tensor<vector<2xf32>>>) -> ()

// -----

// CHECK: !tf.resource
func private @resource_without_type(!tf.resource) -> ()

// CHECK: !tf.resource<tensor<?xf32>>
func private @resource_with_type(!tf.resource<tensor<?xf32>>) -> ()

// CHECK: !tf.resource<tensor<3xf32>, tensor<2xi32>>
func private @resource_with_multiple_types(!tf.resource<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf.resource<tensor<?xf32>>>
func private @resource_element_type(tensor<*x!tf.resource<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf.resource<tensor<?x!tf.resource<tensor<?xf32>>>>>
func private @nested_resource(tensor<!tf.resource<tensor<?x!tf.resource<tensor<?xf32>>>>>) -> ()

// CHECK: !tf.resourceref
func private @resourceref(!tf.resourceref) -> ()

// -----

// expected-error @+1 {{encountered unexpected token}}
func private @invalid_type(!tf<"resource>">) -> ()

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf.resource<>) -> ()

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf.resource<tensor<??xf32>>) -> ()

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf.resource<vector<3xf32>>) -> ()

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf.resource<tensor<vector<2xf32>>>) -> ()

// -----
