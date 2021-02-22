// RUN: kernel-gen-opt %s -split-input-file -verify-diagnostics

func @alloc_raw(%ctx: !tf_framework.op_kernel_context, %size : index) {
  // expected-error @+1 {{`dyn_sizes` count 1 does not match dynamic dimensions}}
  %buf = tf_framework.alloc(%ctx, %size) : memref<?x10x?xi8>
  return
}

// -----

func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) {
  // expected-error @+1{{number of operand shapes 2 does not match number of result shapes 1}}
  %0 = tf_framework.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return
}
