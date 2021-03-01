// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file | mlir-hlo-opt | FileCheck %s

// CHECK-LABEL: func @minimum_broadcast_shapes
func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>)
    -> (tensor<?xindex>, tensor<?xindex>) {
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  return %0, %1 : tensor<?xindex>, tensor<?xindex>
}

// -----

func @minimum_broadcast_shapes_mismatch_operand_and_result_count(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) {
  // expected-error @+1{{number of operand shapes (2) does not match number of result shapes (1)}}
  %0 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return
}

// -----

func @minimum_broadcast_shapes_one_operand(%arg: tensor<?xindex>) {
  // expected-error @+1{{number of operand shapes (1) should be >= 2}}
  %0 = chlo.minimum_broadcast_shapes %arg : tensor<?xindex> -> tensor<?xindex>
  return
}
