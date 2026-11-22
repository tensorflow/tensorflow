// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: litert-opt %s -split-input-file --tfl-lower-quant-annotations | FileCheck %s

// A uniform blockwise zero point, materialized at the full block grid shape,
// is compacted to extent 1 on every axis. Centered int2 with one block per
// row: row 0 holds (q + 0.5) * 0.5 for q = [-2, -1, 0, 1], row 1
// (q + 0.5) * 0.25 for q = [1, 0, -1, -2].

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @uniform_splat_zero_point
func.func @uniform_splat_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK-DAG: %[[S:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>}>
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<-5.000000e-01> : tensor<1x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %[[S]], %[[ZP]])
  %w = stablehlo.constant dense<[[-0.75, -0.25, 0.25, 0.75], [0.375, 0.125, -0.125, -0.375]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<2x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A uniform zero point stored as a (non-splat) dense resource.

// CHECK-LABEL: func.func @uniform_resource_zero_point
func.func @uniform_resource_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<-5.000000e-01> : tensor<1x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-0.75, -0.25, 0.25, 0.75], [0.375, 0.125, -0.125, -0.375]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense_resource<zp> : tensor<2x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      zp: "0x04000000000000BF000000BF"
    }
  }
#-}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A rank 0 zero point is expanded to the rank of the block grid.

// CHECK-LABEL: func.func @scalar_zero_point
func.func @scalar_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<-5.000000e-01> : tensor<1x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-0.75, -0.25, 0.25, 0.75], [0.375, 0.125, -0.125, -0.375]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<f32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A zero point that varies across blocks keeps its shape. Row 0 uses zero
// point 0, row 1 zero point 1: row 1 holds (q - 1) * 0.25 for q = [1, 0, -1].

// CHECK-LABEL: func.func @varying_zero_point
func.func @varying_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -1]]> : tensor<2x4xi2>}>
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[0.000000e+00], [1.000000e+00]]> : tensor<2x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-1.0, -0.5, 0.0, 0.5], [0.0, -0.25, -0.5, -0.5]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<[[0.0], [1.0]]> : tensor<2x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// An already compact zero point is kept as is.

// CHECK-LABEL: func.func @compact_zero_point
func.func @compact_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<-5.000000e-01> : tensor<1x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-0.75, -0.25, 0.25, 0.75], [0.375, 0.125, -0.125, -0.375]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<-0.5> : tensor<1x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A dense resource zero point that varies across blocks keeps its shape.

// CHECK-LABEL: func.func @varying_resource_zero_point
func.func @varying_resource_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense_resource<zp> : tensor<2x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-1.0, -0.5, 0.0, 0.5], [0.0, -0.25, -0.5, -0.5]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense_resource<zp> : tensor<2x1xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      zp: "0x04000000000000000000803F"
    }
  }
#-}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// Integer zero points are compacted the same way.

// CHECK-LABEL: func.func @uniform_integer_zero_point
func.func @uniform_integer_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1x1xi32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense<1> : tensor<2x1xi32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// Only the axes along which the zero point does not vary are collapsed: with
// two blocks per row, a zero point that only varies across rows becomes one
// value per row, and one that only varies within rows one value per column.

// CHECK-LABEL: func.func @per_row_zero_point
func.func @per_row_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[0.000000e+00], [1.000000e+00]]> : tensor<2x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 2]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5, 0.5], [0.25, 0.25]]> : tensor<2x2xf32>, zero_point = dense<[[0.0, 0.0], [1.0, 1.0]]> : tensor<2x2xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @per_column_integer_zero_point
func.func @per_column_integer_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[1, -1]]> : tensor<1x2xi32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 2]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5, 0.5], [0.25, 0.25]]> : tensor<2x2xf32>, zero_point = dense<[[1, -1], [1, -1]]> : tensor<2x2xi32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// A dense resource zero point is compacted into a dense attribute.

// CHECK-LABEL: func.func @per_row_resource_zero_point
func.func @per_row_resource_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[0.000000e+00], [1.000000e+00]]> : tensor<2x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 2]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5, 0.5], [0.25, 0.25]]> : tensor<2x2xf32>, zero_point = dense_resource<zp_rows> : tensor<2x2xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      zp_rows: "0x0400000000000000000000000000803F0000803F"
    }
  }
#-}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A zero point that varies along every axis keeps its shape.

// CHECK-LABEL: func.func @fully_varying_zero_point
func.func @fully_varying_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00]]> : tensor<2x2xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 2]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5, 0.5], [0.25, 0.25]]> : tensor<2x2xf32>, zero_point = dense<[[0.0, 1.0], [1.0, 0.0]]> : tensor<2x2xf32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  return %arg0 : tensor<2x4x4xf32>
}

// A uniform middle axis is collapsed while the varying axes around it are
// kept.

// CHECK-LABEL: func.func @middle_axis_zero_point
func.func @middle_axis_zero_point() -> tensor<2x4x4xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[\[\[}}0, 1]], {{\[\[}}-2, -1]]]> : tensor<2x1x2xi32>}>
  // CHECK: "tfl.blockwise_dequantize"(%{{.+}}, %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<0.0> : tensor<2x4x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 2, 2]> : tensor<3xi64>, symmetric = false, scale = dense<0.5> : tensor<2x2x2xf32>, zero_point = dense<[[[0, 1], [0, 1]], [[-2, -1], [-2, -1]]]> : tensor<2x2x2xi32>}, decomposition = @fq_impl} : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0 : tensor<2x4x4xf32>
}

// -----

func.func private @fq_impl(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

// A rank 0 dense resource zero point is expanded to the rank of the grid.

// CHECK-LABEL: func.func @scalar_resource_zero_point
func.func @scalar_resource_zero_point() -> tensor<2x4xf32> {
  // CHECK-DAG: %[[Q:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[}}[-2, -1, 0, 1], [1, 0, -1, -2]]> : tensor<2x4xi2>}>
  // CHECK-DAG: %[[ZP:.+]] = "tfl.pseudo_const"() <{value = dense_resource<zp_scalar> : tensor<1x1xf32>}>
  // CHECK: "tfl.blockwise_dequantize"(%[[Q]], %{{.+}}, %[[ZP]])
  %w = stablehlo.constant dense<[[-0.75, -0.25, 0.25, 0.75], [0.375, 0.125, -0.125, -0.375]]> : tensor<2x4xf32>
  %0 = stablehlo.composite "quant.fake_quant" %w {composite_attributes = {dtype = "i2", narrow_range = false, block_shape = dense<[1, 4]> : tensor<2xi64>, symmetric = false, scale = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_point = dense_resource<zp_scalar> : tensor<f32>}, decomposition = @fq_impl} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

{-#
  dialect_resources: {
    builtin: {
      zp_scalar: "0x04000000000000BF"
    }
  }
#-}
