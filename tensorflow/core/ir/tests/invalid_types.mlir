// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tfg-opt-no-passes %s -split-input-file -verify-diagnostics | FileCheck %s

// expected-error @+1 {{expected '}'}}
module attributes { tfg.type = !tf_type<variant>> } {}

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid variant type}}
module attributes { tfg.type = !tf_type.variant<>} {}

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid variant type}}
module attributes { tfg.type = !tf_type.variant<tensor<??xf32>>} {}

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid variant type}}
module attributes { tfg.type = !tf_type.variant<vector<3xf32>>} {}

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid variant type}}
module attributes { tfg.type = !tf_type.variant<tensor<vector<2xf32>>>} {}

// -----

// CHECK: !tf_type.resource
module attributes { tfg.type = !tf_type.resource} {}

// CHECK: !tf_type.resource<tensor<?xf32>>
module attributes { tfg.type = !tf_type.resource<tensor<?xf32>>} {}

// CHECK: !tf_type.resource<tensor<3xf32>, tensor<2xi32>>
module attributes { tfg.type = !tf_type.resource<tensor<3xf32>, tensor<2xi32>>} {}

// CHECK: tensor<*x!tf_type.resource<tensor<?xf32>>>
module attributes { tfg.type = tensor<*x!tf_type.resource<tensor<?xf32>>>} {}

// CHECK: tensor<!tf_type.resource<tensor<?x!tf_type.resource<tensor<?xf32>>>>>
module attributes { tfg.type = tensor<!tf_type.resource<tensor<?x!tf_type.resource<tensor<?xf32>>>>>} {}

// CHECK: !tf_type.resourceref
module attributes { tfg.type = !tf_type.resourceref} {}

// -----

// expected-error @+1 {{expected '}'}}
module attributes { tfg.type = !tf_type<resource>>} {}

// -----

// expected-error @+2 {{expected non-function type}}
// expected-error @+1 {{invalid resource type}}
module attributes { tfg.type = !tf_type.resource<>} {}

// -----

// expected-error @+2 {{expected 'x' in dimension list}}
// expected-error @+1 {{invalid resource type}}
module attributes { tfg.type = !tf_type.resource<tensor<??xf32>>} {}

// -----

// expected-error @+2 {{invalid kind of type specified}}
// expected-error @+1 {{invalid resource type}}
module attributes { tfg.type = !tf_type.resource<vector<3xf32>>} {}

// -----

// expected-error @+2 {{invalid subtype: 'tensor<vector<2xf32>>'}}
// expected-error @+1 {{invalid resource type}}
module attributes { tfg.type = !tf_type.resource<tensor<vector<2xf32>>>} {}


// -----

// expected-error @+1 {{expected 'producer' in tf_type version}}
tfg.graph #tf_type.version<> {
}
