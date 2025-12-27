// RUN: xla-opt %s -split-input-file \
// RUN: -stablehlo-lower-to-triton \
// RUN: | FileCheck %s

// -----

func.func @lower_compare_eq_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf oeq, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ne_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf une, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_lt_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf olt, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_gt_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf ogt, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_le_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf ole, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ge_f32(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpf oge, %arg0, %arg1 : tensor<2x4xf32>
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// -----

func.func @lower_compare_eq_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi eq, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ne_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi ne, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_lt_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi slt, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_gt_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi sgt, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_le_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi sle, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ge_i32(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[RES:.*]] = arith.cmpi sge, %arg0, %arg1 : tensor<2x4xi32>
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// -----

func.func @lower_compare_eq_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi eq, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ne_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi ne, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_lt_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi ult, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_gt_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi ugt, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_le_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi ule, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

func.func @lower_compare_ge_ui32(%arg0: tensor<2x4xui32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi1> {
  // CHECK: %[[ARG0_CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[ARG1_CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: %[[RES:.*]] = arith.cmpi uge, %[[ARG0_CAST]], %[[ARG1_CAST]] : tensor<2x4xi32>
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<2x4xui32>, tensor<2x4xui32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}
