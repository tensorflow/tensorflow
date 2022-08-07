// RUN: mlir-hlo-opt --mhlo-test-infer-shaped-type-methods --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @select
// CHECK-SAME: (%{{.*}}: tensor<i1>, %[[SHAPED_ARG:.*]]: tensor<2x?xf32>, %{{.*}}: tensor<2x?xf32>
func.func @select(%pred : tensor<i1>, %a : tensor<2x?xf32>, %b : tensor<2x?xf32>)
    -> tensor<2xindex> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[SHAPED_ARG]] : tensor<2x?xf32> -> tensor<2xindex>
  // CHECK: return %[[SHAPE]] : tensor<2xindex>
  %0 = "mhlo.select"(%pred, %a, %b)
      : (tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%0)
      : (tensor<2x?xf32>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @compare
// CHECK-SAME: (%[[A:.*]]: tensor<2x?xf32>,
func.func @compare(%a : tensor<2x?xf32>, %b : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[A]] : tensor<2x?xf32> -> tensor<2xindex>
  // CHECK: return %[[SHAPE]] : tensor<2xindex>
  %0 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<comparison_direction NE>}
      : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xi1>
  %1 = "mhlo_test.reify_return_type_shapes"(%0)
      : (tensor<2x?xi1>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @select
func.func @select(%pred : tensor<i1>, %a : tensor<2x2xf32>, %b : tensor<2x2xf32>)
    -> tensor<2x2xindex> {
  %0 = "mhlo.select"(%pred, %a, %b)
      : (tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<2x2xf32>) -> tensor<2x2xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [2, 2], element_type0 = f32} : (tensor<2x2xf32>) -> tensor<2x2xindex>
  func.return %1 : tensor<2x2xindex>
}

// -----

// CHECK-LABEL: @compare
func.func @compare(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xindex> {
  %0 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<comparison_direction NE>}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<2x2xi1>) -> tensor<2x2xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [2, 2], element_type0 = i1} : (tensor<2x2xi1>) -> tensor<2x2xindex>
  func.return %1 : tensor<2x2xindex>
}

// -----

// CHECK-LABEL: @broadcast
func.func @broadcast(%a : tensor<3xi32>) -> tensor<1x2x3xindex> {
  %0 = "mhlo.broadcast"(%a) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>}
      : (tensor<3xi32>) -> tensor<1x2x3xi32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1x2x3xi32>) -> tensor<1x2x3xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [1, 2, 3], element_type0 = i32} : (tensor<1x2x3xi32>) -> tensor<1x2x3xindex>
  func.return %1 : tensor<1x2x3xindex>
}

// -----

func.func @broadcast(%a : tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{Broadcast with negative dimension size -2}}
  %0 = "mhlo.broadcast"(%a) {broadcast_sizes = dense<[1, -2]> : tensor<2xi64>}
      : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xindex> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1x4xi32>) -> tensor<1x4xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [1, 4], element_type0 = i32} : (tensor<1x4xi32>) -> tensor<1x4xindex>
  func.return %1 : tensor<1x4xindex>
}

// -----

// CHECK-LABEL: @pad
func.func @pad(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<2x4x7xf16>) -> tensor<2x4x7xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [2, 4, 7], element_type0 = f16} : (tensor<2x4x7xf16>) -> tensor<2x4x7xindex>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

// CHECK-LABEL: @cholesky
func.func @cholesky(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2xindex> {
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1x2x2xf32>) -> tensor<1x2x2xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [1, 2, 2], element_type0 = f32} : (tensor<1x2x2xf32>) -> tensor<1x2x2xindex>
  func.return %1: tensor<1x2x2xindex>
}

// -----

// CHECK-LABEL: func @alltoall
func.func @alltoall(%data: tensor<4x16xf32>) -> tensor<16x4xindex> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<16x4xf32>) -> tensor<16x4xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [16, 4], element_type0 = f32} : (tensor<16x4xf32>) -> tensor<16x4xindex>
  func.return %1 : tensor<16x4xindex>
}

// -----

// CHECK-LABEL: func @abs
func.func @abs(%arg0: tensor<1x2xf32>) -> tensor<1x2xindex> {
  %0 = "mhlo.abs"(%arg0) {} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1x2xf32>) -> tensor<1x2xindex>
// CHECK: %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<1x2xf32>) -> tensor<1x2xindex>
  func.return %1: tensor<1x2xindex>
}

// -----

// CHECK-LABEL: @concat
func.func @concat(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xindex> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<3xi32>) -> tensor<3xindex>
// CHECK: %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<3xi32>) -> tensor<3xindex>
  func.return %1 : tensor<3xindex>
}

// -----

// CHECK-LABEL: @gather
func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xindex> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  %1 = "mhlo_test.get_return_type_components"(%res)
      : (tensor<1x5x8xi32>) -> tensor<1x5x8xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [1, 5, 8], element_type0 = i32} : (tensor<1x5x8xi32>) -> tensor<1x5x8xindex>
  func.return %1 : tensor<1x5x8xindex>
}

// -----

// CHECK-LABEL: @rng_normal
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<7xindex> {
  %0 = "mhlo.constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "mhlo.rng"(%arg0, %arg1, %0) {rng_distribution = #mhlo.rng_distribution<NORMAL>} : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<7xf32>
  %2 = "mhlo_test.get_return_type_components"(%1)
      : (tensor<7xf32>) -> tensor<7xindex>
// CHECK: %2 = "mhlo_test.return_type_components"(%1) {dims0 = [7], element_type0 = f32} : (tensor<7xf32>) -> tensor<7xindex>
  func.return %2 : tensor<7xindex>
}

// -----

// CHECK-LABEL: func @rng_uniform
func.func @rng_uniform(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xindex> {
  %0 = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %1 = "mhlo.rng"(%a, %b, %0) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  %2 = "mhlo_test.get_return_type_components"(%1)
      : (tensor<2x3x5xf32>) -> tensor<2x3x5xindex>
// CHECK: %2 = "mhlo_test.return_type_components"(%1) {dims0 = [2, 3, 5], element_type0 = f32} : (tensor<2x3x5xf32>) -> tensor<2x3x5xindex>
  func.return %2 : tensor<2x3x5xindex>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xindex> {
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1x2xi32>) -> tensor<1x2xindex>
// CHECK: %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<1x2xi32>) -> tensor<1x2xindex>
  func.return %1 : tensor<1x2xindex>
}

// -----

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1xindex> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<1xi32>) -> tensor<1xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [1], element_type0 = i32} : (tensor<1xi32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----

// CHECK: func @uniform_dequantize
func.func @uniform_dequantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xindex> {
  %0 = mhlo.uniform_dequantize(%arg) : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<16x16xf32>) -> tensor<16x16xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [16, 16], element_type0 = f32} : (tensor<16x16xf32>) -> tensor<16x16xindex>
  func.return %1 : tensor<16x16xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @tanh_sparsity
func.func @tanh_sparsity(%arg0: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.tanh"(%arg0) : (tensor<10x10xf32, #CSR>) -> tensor<10x10xf32>
  %1 = "mhlo_test.get_return_types"(%0)
      : (tensor<10x10xf32>) -> tensor<10x10xindex>
// CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = tensor<10x10xf32, {{.*}}>} : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @abs_sparsity
func.func @abs_sparsity(%arg0: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.abs"(%arg0) : (tensor<10x10xf32, #CSR>) -> tensor<10x10xf32>
  %1 = "mhlo_test.get_return_types"(%0)
      : (tensor<10x10xf32>) -> tensor<10x10xindex>
// CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = tensor<10x10xf32, {{.*}}>} : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @real_sparsity
func.func @real_sparsity(%arg0: tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.real"(%arg0) : (tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xf32>
  %1 = "mhlo_test.get_return_types"(%0)
      : (tensor<10x10xf32>) -> tensor<10x10xindex>
// CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = tensor<10x10xf32, {{.*}}>} : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @imag_sparsity
func.func @imag_sparsity(%arg0: tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.imag"(%arg0) : (tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xf32>
  %1 = "mhlo_test.get_return_types"(%0)
      : (tensor<10x10xf32>) -> tensor<10x10xindex>
// CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = tensor<10x10xf32, {{.*}}>} : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @complex_sparsity
func.func @complex_sparsity(%arg0: tensor<10x10xf32, #CSR>, %arg1: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<10x10xf32, #CSR>, tensor<10x10xf32, #CSR>) -> tensor<10x10xcomplex<f32>>
  %1 = "mhlo_test.get_return_types"(%0)
      : (tensor<10x10xcomplex<f32>>) -> tensor<10x10xindex>
// CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = tensor<10x10xcomplex<f32>, {{.*}}>} : (tensor<10x10xcomplex<f32>>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

// CHECK-LABEL: @tensor_bounds
func.func @tensor_bounds(%arg0: tensor<3x5xf32>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x5xf32>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @edit_tensor_bounds
func.func @edit_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, 5]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @retain_tensor_bounds
func.func @retain_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, -1]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @unknown_bounds
func.func @unknown_bounds(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @unranked_input
func.func @unranked_input(%arg0: tensor<*xf32>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<*xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// This test covers all cases (except the "Error out" case) for type inference
// of binary op with bounds
// See PairwiseSameOperandAndResultType::inferDimWithBound()
// CHECK-LABEL: @add_bounds
func.func @add_bounds(
  %arg0: tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, -1, -1, -1, 3, 3]>>,
  %arg1: tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, 4, -1, 3, 3, 4]>>) -> tensor<*xindex> {
  %result1 = "mhlo.add"(%arg0, %arg1) : (
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, -1, -1, -1, 3, 3]>>,
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, 4, -1, 3, 3, 4]>>)
    -> tensor<?x?x?x?x?x?x?xf32>
  %result2 = "mhlo.add"(%arg1, %arg0) : (
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, 4, -1, 3, 3, 4]>>,
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, -1, -1, -1, 3, 3]>>)
    -> tensor<?x?x?x?x?x?x?xf32>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, -1, -1, 3, 3, 3]>>
  %1 = "mhlo_test.get_return_types"(%result1) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<*xindex>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [-1, -1, -1, -1, 3, 3, 3]>>
  %2 = "mhlo_test.get_return_types"(%result2) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// This test covers "Error out" case for type inference of binary op with bounds
// See PairwiseSameOperandAndResultType::inferDimWithBound()
func.func @add_bounds_mismatch(
  %arg0: tensor<3xf32, #mhlo.type_extensions<bounds = [-1]>>,
  %arg1: tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> tensor<*xindex> {
  // expected-error@+1 {{bound must not be less than static dimension size but has bound 2 vs static size 3}}
  %result = "mhlo.add"(%arg0, %arg1) : (
    tensor<3xf32, #mhlo.type_extensions<bounds = [-1]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> tensor<?xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @add_bounds_unranked
func.func @add_bounds_unranked(
  %arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xindex> {
  %result = "mhlo.add"(%arg0, %arg1) : (
    tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: types0 = tensor<*xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}