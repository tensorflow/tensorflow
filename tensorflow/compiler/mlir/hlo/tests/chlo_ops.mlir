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

// -----

func @rank_specialization_cluster(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
    %1 = chlo.broadcast_multiply %arg0_, %arg1_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = chlo.broadcast_add %1, %arg2_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "chlo.rank_specialization_cluster_yield"(%2) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @rank_specialization_cluster(%arg0 : tensor<*xf32>,
    %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{source has 2 operands, but target successor needs 1}}
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>):
    "chlo.rank_specialization_cluster_yield"(%arg0_, %arg1_)
        : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @rank_specialization_cluster(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{block argument types must match operand types}}
  %0 = "chlo.rank_specialization_cluster"(%arg0) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>):
    "chlo.rank_specialization_cluster_yield"(%arg0_) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @rank_specialization_cluster(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1{{nested ops must not depend on implicit operands}}
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
    %1 = chlo.broadcast_multiply %arg0_, %arg1_
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = chlo.broadcast_add %1, %arg2
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "chlo.rank_specialization_cluster_yield"(%2) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
