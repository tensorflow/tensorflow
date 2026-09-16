// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt -verify-diagnostics                                             \
// RUN:        -allow-unregistered-dialect                                     \
// RUN:        -tf-test-clustering-policy %s                                   \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @propagate_constraints
func.func @propagate_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "test.OpA"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  %1 = "test.OpB"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @failed_to_propagate_constraints
func.func @failed_to_propagate_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints: 0:value}}
  %0 = "test.OpC"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  func.return %0 : tensor<?x?xf32>
}
