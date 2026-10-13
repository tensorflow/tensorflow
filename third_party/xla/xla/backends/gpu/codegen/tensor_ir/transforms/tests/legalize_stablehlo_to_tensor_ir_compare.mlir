// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// The twelve rows of the comparison-direction mapping. Float directions map to
// the ordered comparators except NE, which maps to the unordered `une` because
// IEEE 754 defines `NaN != x` as true.

// CHECK-LABEL: nv_tensor_ir.graph @compare_float_eq
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] oeq %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_eq(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// NE is the one asymmetric row: `une`, not `one`.
// CHECK-LABEL: nv_tensor_ir.graph @compare_float_ne
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] une %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_ne(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_float_gt
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] ogt %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_gt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_float_ge
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] oge %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_ge(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_float_lt
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] olt %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_lt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_float_le
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] ole %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float_le(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_eq
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] eq %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_eq(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_ne
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] neq %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_ne(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_gt
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] gt %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_gt(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_ge
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] ge %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_ge(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_lt
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] lt %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_lt(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_integer_le
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] le %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_integer_le(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
