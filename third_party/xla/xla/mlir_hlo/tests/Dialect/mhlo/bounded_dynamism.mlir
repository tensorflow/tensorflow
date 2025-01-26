// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// This file captures some quirks to bounded dynamism in MHLO.
// HLO treats bounded dimensions as essentially static.


// CHECK-LABEL: reshape_with_single_bounded_dimension
func.func @reshape_with_single_bounded_dimension(%arg0: tensor<?x2xf32, #mhlo.type_extensions<bounds = [5, ?]>>) -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 5]>> {
  %0 = mhlo.reshape %arg0 : (tensor<?x2xf32, #mhlo.type_extensions<bounds = [5, ?]>>) -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  // CHECK: return {{.*}} #mhlo.type_extensions<bounds = [?, 5]
  return %0 : tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
}

// -----

func.func @reshape_with_multiple_bounded_dimensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>> {
  // expected-error@+1 {{load bearing ops with dynamic dimensions must have a single bounded dimension}}
  %0 = mhlo.reshape %arg0 : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>
  return %0 : tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>
}

// -----

// CHECK-LABEL: broadcast_in_dim_with_single_bounded_dimension
func.func @broadcast_in_dim_with_single_bounded_dimension(%arg0: tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>) -> tensor<2x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 5]>> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>) -> tensor<2x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 5]>>
  // CHECK: return {{.*}} #mhlo.type_extensions<bounds = [?, ?, 5]
  return %0 : tensor<2x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 5]>>
}

// -----

func.func @broadcast_in_dim_with_multiple_bounded_dimensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, 5]>> {
  // expected-error@+1 {{load bearing ops with dynamic dimensions must have a single bounded dimension}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, 5]>>
  return %0 : tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, 5]>>
}
