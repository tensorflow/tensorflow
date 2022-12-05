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
  %0 = mhlo.uniform_dequantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<16x16xf32>) -> tensor<16x16xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [16, 16], element_type0 = f32} : (tensor<16x16xf32>) -> tensor<16x16xindex>
  func.return %1 : tensor<16x16xindex>
}

// -----

// CHECK-LABEL: func @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xindex> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  %1 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [3, 9], element_type0 = complex<f32>} : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xindex>
  func.return %1 : tensor<3x9xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_grad
func.func @batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xindex> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  // CHECK: (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xindex>
  %1 = "mhlo_test.get_return_type_components"(%0#0) : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xindex>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %2 = "mhlo_test.get_return_type_components"(%0#1) : (tensor<2xf32>) -> tensor<2xindex>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %3 = "mhlo_test.get_return_type_components"(%0#2) : (tensor<2xf32>) -> tensor<2xindex>
  func.return %1 : tensor<2x2x2x2xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_train
func.func @batch_norm_train(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xindex> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 1 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  // CHECK: (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xindex>
  %1 = "mhlo_test.get_return_type_components"(%0#0) : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xindex>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %2 = "mhlo_test.get_return_type_components"(%0#1) : (tensor<2xf32>) -> tensor<2xindex>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %3 = "mhlo_test.get_return_type_components"(%0#2) : (tensor<2xf32>) -> tensor<2xindex>
  func.return %1 : tensor<2x2x2x2xindex>
}

// -----

// CHECK-LABEL: @batch_norm_inference
func.func @batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xindex>) {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK: (tensor<4x256xf32>) -> tensor<4x256xindex>
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<4x256xf32>) -> tensor<4x256xindex>
  func.return %1 : tensor<4x256xindex>
}

// -----

// CHECK-LABEL: func @map
func.func @map(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xindex> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK: (tensor<4x5xf32>) -> tensor<4x5xindex>
  %2 = "mhlo_test.get_return_type_components"(%0) : (tensor<4x5xf32>) -> tensor<4x5xindex>
  func.return %2 : tensor<4x5xindex>
}

// -----

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xindex> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  // CHECK: (tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xindex>
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xindex>
  func.return %1 : tensor<10x5x4x4xindex>
}

// -----

// CHECK-LABEL: func @if
func.func @if(%pred : tensor<i1>, %branch_operand : tensor<2xf32>, %wrong_type : tensor<2xf32>) {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%wrong_type) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<2xf32>) -> tensor<2xindex>
  func.return
}

// -----

// CHECK-LABEL: func @case
func.func @case(%index : tensor<i32>, %branch_operand : tensor<2xf32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<i32>) -> tensor<2xf32>
  // CHECK: (tensor<2xf32>) -> tensor<2xindex>
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<2xf32>) -> tensor<2xindex>
  func.return
}

// -----

// CHECK-LABEL: func @sort
func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  // CHECK: (tensor<16x16xf32>) -> tensor<16x16xindex>
  %1 = "mhlo_test.get_return_type_components"(%0#0) : (tensor<16x16xf32>) -> tensor<16x16xindex>
  // CHECK: (tensor<16x16xi32>) -> tensor<16x16xindex>
  %2 = "mhlo_test.get_return_type_components"(%0#1) : (tensor<16x16xi32>) -> tensor<16x16xindex>
  func.return
}

// -----

// CHECK-LABEL: func @while
func.func @while(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>) -> tensor<index> {
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %cst_1 = arith.constant dense<0> : tensor<i32>
  %cst_2 = arith.constant dense<1000> : tensor<i32>
  %1:3 = "mhlo.while"(%cst_1, %cst, %cst_2) ({
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %4 = "mhlo.compare"(%arg9, %arg11) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %3 = mhlo.add %arg9, %cst_0 : tensor<i32>
    "mhlo.return"(%3, %arg10, %arg11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: (tensor<i32>) -> tensor<index>
  %4 = "mhlo_test.get_return_type_components"(%1#0) : (tensor<i32>) -> tensor<index>
  // CHECK: (tensor<i32>) -> tensor<index>
  %5 = "mhlo_test.get_return_type_components"(%1#1) : (tensor<i32>) -> tensor<index>
  // CHECK: (tensor<i32>) -> tensor<index>
  %6 = "mhlo_test.get_return_type_components"(%1#2) : (tensor<i32>) -> tensor<index>
  func.return %4 : tensor<index>
}

// -----

//===----------------------------------------------------------------------===//
// Sparsity
//===----------------------------------------------------------------------===//

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

// CHECK-LABEL: func @reduce
func.func @reduce(%arg0: tensor<4x4xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<4xindex>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo_test.get_return_type_components"(%0)
      : (tensor<4xf32>) -> tensor<4xindex>
// CHECK: %1 = "mhlo_test.return_type_components"(%0) {dims0 = [4], element_type0 = f32} : (tensor<4xf32>) -> tensor<4xindex>
  func.return %2: tensor<4xindex>
}

// -----

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) ->
                      (tensor<2x2xindex>, tensor<2x2xindex>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  // CHECK: %1 = "mhlo_test.return_type_components"(%0#0) {dims0 = [2, 2], dims1 = [2, 2], element_type0 = f32, element_type1 = i32} : (tensor<2x2xf32>) -> tensor<2x2xindex>
  %1 = "mhlo_test.get_return_type_components"(%0#0)
      : (tensor<2x2xf32>) -> tensor<2x2xindex>
  // CHECK: "mhlo_test.return_type_components"(%0#1) {dims0 = [2, 2], dims1 = [2, 2], element_type0 = f32, element_type1 = i32} : (tensor<2x2xi32>) -> tensor<2x2xindex>
  %2 = "mhlo_test.get_return_type_components"(%0#1)
      : (tensor<2x2xi32>) -> tensor<2x2xindex>
  func.return %1, %2 : tensor<2x2xindex>, tensor<2x2xindex>
}

// -----

//===----------------------------------------------------------------------===//
// Bounded Dynamism
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tensor_bounds
func.func @tensor_bounds(%arg0: tensor<3x5xf32>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x5xf32>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @static_tensor_bounds
func.func @static_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [8, ?]>>) -> tensor<*xindex> {
  %bounds = mhlo.constant dense<8> : tensor<i32>
  %result = "mhlo.set_dimension_size"(%arg0, %bounds) {dimension = 0 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [8, ?]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<8x5xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @edit_tensor_bounds
func.func @edit_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, 5]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @retain_tensor_bounds
func.func @retain_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<*xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// CHECK-LABEL: @unknown_bounds
func.func @unknown_bounds(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<*xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<*xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>
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
  %arg0: tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>,
  %arg1: tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>) -> tensor<*xindex> {
  %result1 = "mhlo.add"(%arg0, %arg1) : (
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>,
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>)
    -> tensor<?x?x?x?x?x?x?xf32>
  %result2 = "mhlo.add"(%arg1, %arg0) : (
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>,
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>)
    -> tensor<?x?x?x?x?x?x?xf32>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, 3, 3, 3]>>
  %1 = "mhlo_test.get_return_types"(%result1) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<*xindex>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, 3, 3, 3]>>
  %2 = "mhlo_test.get_return_types"(%result2) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}

// -----

// This test covers "Error out" case for type inference of binary op with bounds
// See PairwiseSameOperandAndResultType::inferDimWithBound()
func.func @add_bounds_mismatch(
  %arg0: tensor<3xf32, #mhlo.type_extensions<bounds = [?]>>,
  %arg1: tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> tensor<*xindex> {
  // expected-error@+1 {{requires compatible types for all operands and results}}
  %result = "mhlo.add"(%arg0, %arg1) : (
    tensor<3xf32, #mhlo.type_extensions<bounds = [?]>>,
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

// -----

// CHECK-LABEL: @partition_id
func.func @partition_id() -> tensor<*xindex> {
  %result = "mhlo.partition_id"() : () -> tensor<ui32>
  // CHECK: types0 = tensor<ui32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<ui32>) -> tensor<*xindex>
  func.return %1 : tensor<*xindex>
}
