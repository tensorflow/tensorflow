// RUN: fusion_compiler_opt %s -xtile-cpu-drop-vector-unit-dims \
// RUN:   -split-input-file | FileCheck %s

func.func @drop_trailing_unit_dim(%arg0: vector<16x1xf32>) -> vector<16x1xf32> {
  %0 = math.isfinite %arg0 : vector<16x1xf32>
  %1 = arith.extui %0 : vector<16x1xi1> to vector<16x1xi8>
  %2 = arith.uitofp %1 : vector<16x1xi8> to vector<16x1xf32>
  return %2 : vector<16x1xf32>
}
// CHECK-LABEL: func.func @drop_trailing_unit_dim(
// CHECK-SAME:     %[[ARG0:.*]]: vector<16x1xf32>) -> vector<16x1xf32> {
// CHECK: %[[CAST_IN:.*]] = vector.shape_cast %[[ARG0]] : vector<16x1xf32> to vector<16xf32>
// CHECK: %[[FINITE:.*]] = math.isfinite %[[CAST_IN]] : vector<16xf32>
// CHECK: %[[EXTUI:.*]] = arith.extui %[[FINITE]] : vector<16xi1> to vector<16xi8>
// CHECK: %[[UITOFP:.*]] = arith.uitofp %[[EXTUI]] : vector<16xi8> to vector<16xf32>
// CHECK: vector.shape_cast %[[UITOFP]] : vector<16xf32> to vector<16x1xf32>

// -----

func.func @drop_leading_unit_dim(%arg0: vector<1x16xf32>, %arg1: vector<1x16xf32>) -> vector<1x16xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<1x16xf32>
  return %0 : vector<1x16xf32>
}
// CHECK-LABEL: func.func @drop_leading_unit_dim(
// CHECK-SAME:     %[[ARG0:.*]]: vector<1x16xf32>, %[[ARG1:.*]]: vector<1x16xf32>) -> vector<1x16xf32> {
// CHECK: %[[CAST_A:.*]] = vector.shape_cast %[[ARG0]] : vector<1x16xf32> to vector<16xf32>
// CHECK: %[[CAST_B:.*]] = vector.shape_cast %[[ARG1]] : vector<1x16xf32> to vector<16xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[CAST_A]], %[[CAST_B]] : vector<16xf32>
// CHECK: vector.shape_cast %[[ADD]] : vector<16xf32> to vector<1x16xf32>

// -----

func.func @drop_leading_and_trailing_unit_dims(%arg0: vector<1x16x1xf32>) -> vector<1x16x1xf32> {
  %0 = math.absf %arg0 : vector<1x16x1xf32>
  return %0 : vector<1x16x1xf32>
}
// CHECK-LABEL: func.func @drop_leading_and_trailing_unit_dims(
// CHECK-SAME:     %[[ARG0:.*]]: vector<1x16x1xf32>) -> vector<1x16x1xf32> {
// CHECK: %[[CAST_IN:.*]] = vector.shape_cast %[[ARG0]] : vector<1x16x1xf32> to vector<16xf32>
// CHECK: %[[ABS:.*]] = math.absf %[[CAST_IN]] : vector<16xf32>
// CHECK: vector.shape_cast %[[ABS]] : vector<16xf32> to vector<1x16x1xf32>

// -----

func.func @no_unit_dims(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: func.func @no_unit_dims(
// CHECK-SAME:     %[[ARG0:.*]]: vector<16xf32>, %[[ARG1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NOT: vector.shape_cast
// CHECK:     arith.addf %[[ARG0]], %[[ARG1]] : vector<16xf32>
