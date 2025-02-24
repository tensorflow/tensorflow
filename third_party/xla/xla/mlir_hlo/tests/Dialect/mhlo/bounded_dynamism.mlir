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

// CHECK-LABEL: reshape_scalar_with_single_bounded_dimension
func.func @reshape_scalar_with_single_bounded_dimension(%arg0: tensor<?xf32, #mhlo.type_extensions<bounds = [5]>>) -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>> {
  %0 = mhlo.reshape %arg0 : (tensor<?xf32, #mhlo.type_extensions<bounds = [5]>>) -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  // CHECK: return {{.*}} #mhlo.type_extensions<bounds = [?, 5]
  return %0 : tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
}

// -----

func.func @reshape_with_multiple_bounded_dimensions(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>> {
  // expected-error@+1 {{result #0 must be statically shaped or single bounded dimension tensor}}
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
  // expected-error@+1 {{result #0 must be statically shaped or single bounded dimension tensor}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [5, 5]>>) -> tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, 5]>>
  return %0 : tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, 5]>>
}

// -----

// CHECK-LABEL: constant_splat_broadcast
func.func @constant_splat_broadcast() -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>> {
  %0 = mhlo.constant dense<1.0> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  // CHECK: tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  return %1 : tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
}

// -----

func.func @constant_with_dynamic_shape() -> tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>> {
  // expected-error@below {{elements literal type must have static shape}}
  %c = mhlo.constant dense<1> : tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
  return %c : tensor<1x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>
}
