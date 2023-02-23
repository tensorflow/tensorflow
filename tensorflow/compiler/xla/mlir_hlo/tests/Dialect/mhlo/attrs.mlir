// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: parameter_replication
func.func @parameter_replication(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false, true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_empty(%arg0: tensor<f32> {mhlo.parameter_replication = []}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = []}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_single_false(%arg0: tensor<f32> {mhlo.parameter_replication = [false]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: parameter_replication
func.func @parameter_replication_single_true(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}


// -----

// expected-error@+1 {{parameter_replication: arg 0 has 1 leaf_buffers, but parameter_replication expects 2}}
func.func @parameter_replication_num_leaf_buffer_mismatch(%arg0: tensor<f32> {mhlo.parameter_replication = [true, false]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}