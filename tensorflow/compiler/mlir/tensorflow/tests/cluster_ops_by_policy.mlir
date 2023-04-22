// RUN: tf-opt -verify-diagnostics                                             \
// RUN:        -allow-unregistered-dialect                                     \
// RUN:        -tf-test-clustering-policy %s                                   \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @propagate_constraints
func @propagate_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-remark@below {{operand #0 constrained to: rank}}
  %0 = "test.OpA"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: shape}}
  %1 = "test.OpB"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @failed_to_propagate_constraints
func @failed_to_propagate_constraints(%arg0 : tensor<?x?xf32>)
    -> (tensor<?x?xf32> { tf.constraint = "value"  }) {
  // expected-error@below {{failed to propagate results constraints}}
  %0 = "test.OpC"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // expected-remark@below {{operand #0 constrained to: value}}
  return %0 : tensor<?x?xf32>
}
