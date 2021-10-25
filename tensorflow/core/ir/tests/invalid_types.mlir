// RUN: tfg-opt-no-passes %s -split-input-file -verify-diagnostics | FileCheck %s

// expected-error @+1 {{unexpected token}}
func private @invalid_type(!tf_type<"variant>">) -> ()

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf_type.variant<>) -> ()

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf_type.variant<tensor<??xf32>>) -> ()

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf_type.variant<vector<3xf32>>) -> ()

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid variant type}}
func private @invalid_type(!tf_type.variant<tensor<vector<2xf32>>>) -> ()

// -----

// CHECK: !tf_type.resource
func private @resource_without_type(!tf_type.resource) -> ()

// CHECK: !tf_type.resource<tensor<?xf32>>
func private @resource_with_type(!tf_type.resource<tensor<?xf32>>) -> ()

// CHECK: !tf_type.resource<tensor<3xf32>, tensor<2xi32>>
func private @resource_with_multiple_types(!tf_type.resource<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf_type.resource<tensor<?xf32>>>
func private @resource_element_type(tensor<*x!tf_type.resource<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf_type.resource<tensor<?x!tf_type.resource<tensor<?xf32>>>>>
func private @nested_resource(tensor<!tf_type.resource<tensor<?x!tf_type.resource<tensor<?xf32>>>>>) -> ()

// CHECK: !tf_type.resourceref
func private @resourceref(!tf_type.resourceref) -> ()

// -----

// expected-error @+1 {{encountered unexpected token}}
func private @invalid_type(!tf_type<"resource>">) -> ()

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf_type.resource<>) -> ()

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf_type.resource<tensor<??xf32>>) -> ()

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf_type.resource<vector<3xf32>>) -> ()

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid resource type}}
func private @invalid_type(!tf_type.resource<tensor<vector<2xf32>>>) -> ()


// -----

// expected-error @+2 {{expected 'producer' in tf_type version}}
// expected-error @+1 {{expected a version attribute}}
tfg.graph #tf_type.version<> {
}
