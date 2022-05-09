// RUN: mlir-hlo-opt -mhlo-legalize-control-flow %s -o - | FileCheck %s

// CHECK-LABEL: func @while(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1xi64>) -> tensor<1xi64> {
func.func @while(%arg0: tensor<1xi64>) -> tensor<1xi64> {

  // CHECK: %[[VAL_1:.*]] = scf.while (%[[VAL_2:.*]] = %[[VAL_0]]) : (tensor<1xi64>) -> tensor<1xi64> {
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<1xi64>):

    // CHECK: %[[VAL_3:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_2]]) {comparison_direction = #mhlo<"comparison_direction LT">, name = "compare.2"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    // CHECK: %[[RESHAPE:.*]] = "mhlo.reshape"(%[[VAL_3]]) : (tensor<1xi1>) -> tensor<i1>
    // CHECK: %[[VAL_4:.*]] = tensor.extract %[[RESHAPE]][] : tensor<i1>
    // CHECK: scf.condition(%[[VAL_4]]) %[[VAL_2]] : tensor<1xi64>
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">, name = "compare.2"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()

  // CHECK: } do {
  // CHECK: ^bb0(%[[VAL_5:.*]]: tensor<1xi64>):
  },  {
  ^bb0(%arg1: tensor<1xi64>):

    // CHECK: %[[VAL_6:.*]] = mhlo.add %[[VAL_5]], %[[VAL_5]] {name = "compare.0"} : tensor<1xi64>
    // CHECK: scf.yield %[[VAL_6]] : tensor<1xi64>
    %1 = mhlo.add %arg1, %arg1 {name = "compare.0"} : tensor<1xi64>
    "mhlo.return"(%1) : (tensor<1xi64>) -> ()
  }) : (tensor<1xi64>) -> tensor<1xi64>

  // CHECK: return %[[VAL_7:.*]] : tensor<1xi64>
  func.return %0 : tensor<1xi64>
}


// CHECK-LABEL: func @while_multi_operands(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>> {
func.func @while_multi_operands(%arg0: tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>> {

  // CHECK-NEXT: %[[VAL_1:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT: %[[VAL_2:.*]] = mhlo.constant dense<0> : tensor<i32>
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = mhlo.constant dense<0> : tensor<i32>

  // CHECK: %[[VAL_3:.*]]:2 = scf.while (%[[VAL_4:.*]] = %[[VAL_2]], %[[VAL_5:.*]] = %[[VAL_0]]) : (tensor<i32>, tensor<3xi32>) -> (tensor<i32>, tensor<3xi32>) {
  %2:2 = "mhlo.while"(%1, %arg0) ({
  ^bb0(%arg1: tensor<i32> , %arg2: tensor<3xi32> ):

    // CHECK-NEXT: %[[VAL_6:.*]] = mhlo.constant dense<false> : tensor<i1>
    // CHECK-NEXT: %[[VAL_7:.*]] = mhlo.constant dense<8> : tensor<i32>
    // CHECK: %[[VAL_8:.*]] = "mhlo.compare"(%[[VAL_4]], %[[VAL_7]]) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: %[[VAL_9:.*]] = tensor.extract %[[VAL_8]][] : tensor<i1>
    // CHECK: scf.condition(%[[VAL_9]]) %[[VAL_4]], %[[VAL_5]] : tensor<i32>, tensor<3xi32>
    %4 = mhlo.constant dense<false> : tensor<i1>
    %5 = mhlo.constant dense<8> : tensor<i32>
    %6 = "mhlo.compare"(%arg1, %5) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%6) : (tensor<i1>) -> ()
  },  {

  // CHECK: } do {
  // CHECK: ^bb0(%[[VAL_10:.*]]: tensor<i32>, %[[VAL_11:.*]]: tensor<3xi32>):
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xi32>):

    // CHECK-NEXT: %[[VAL_12:.*]] = mhlo.constant dense<false> : tensor<i1>
    // CHECK-NEXT: %[[VAL_13:.*]] = mhlo.constant dense<1> : tensor<i32>
    // CHECK: %[[VAL_14:.*]] = mhlo.add %[[VAL_10]], %[[VAL_13]] : tensor<i32>
    // CHECK: %[[VAL_15:.*]] = mhlo.convert %[[VAL_10]] : tensor<i32>
    // CHECK: %[[VAL_16:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_15]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<3xi32>
    // CHECK: %[[VAL_17:.*]] = mhlo.add %[[VAL_11]], %[[VAL_16]] : tensor<3xi32>
    // CHECK: scf.yield %[[VAL_14]], %[[VAL_17]] : tensor<i32>, tensor<3xi32>
    %4 = mhlo.constant dense<false> : tensor<i1>
    %5 = mhlo.constant dense<1> : tensor<i32>
    %6 = mhlo.add %arg1, %5 : tensor<i32>
    %7 = mhlo.convert(%arg1) : (tensor<i32>) -> tensor<i32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<3xi32>
    %9 = mhlo.add %arg2, %8 : tensor<3xi32>
    "mhlo.return"(%6, %9) : (tensor<i32>, tensor<3xi32>) -> ()
  }) : (tensor<i32>, tensor<3xi32>) -> (tensor<i32>, tensor<3xi32>)

  // CHECK: %[[VAL_18:.*]] = "mhlo.tuple"(%[[VAL_19:.*]]#0, %[[VAL_19]]#1) {xla_shape = "(s32[], s32[3]{0})"} : (tensor<i32>, tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>>
  // CHECK: return %[[VAL_18]] : tuple<tensor<i32>, tensor<3xi32>>
  %3 = "mhlo.tuple"(%2#0, %2#1) {xla_shape = "(s32[], s32[3]{0})"} : (tensor<i32>, tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>>
  func.return %3 : tuple<tensor<i32>, tensor<3xi32>>
}

// CHECK-LABEL: func @conditional(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
func.func @conditional(%arg0: tensor<f32>) -> tensor<f32> {

  // CHECK-NEXT: %[[VAL_1:.*]] = arith.constant dense<1.000000e+01> : tensor<f32>
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>

  // CHECK: %[[VAL_2:.*]] = "mhlo.compare"(%[[VAL_0]], %[[VAL_1]]) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: %[[VAL_3:.*]] = tensor.extract %[[VAL_2]][] : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (tensor<f32>) {
  %1 = "mhlo.if"(%0) ({

    // CHECK: %[[VAL_5:.*]] = mhlo.log %[[VAL_0]] : tensor<f32>
    // CHECK: scf.yield %[[VAL_5]] : tensor<f32>
    %2 = mhlo.log(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()

  // CHECK: } else {
  },  {

    // CHECK: %[[VAL_6:.*]] = mhlo.exponential %[[VAL_0]] : tensor<f32>
    // CHECK: scf.yield %[[VAL_6]] : tensor<f32>
    %2 = mhlo.exponential(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:           return %[[VAL_7:.*]] : tensor<f32>
  func.return %1 : tensor<f32>
}

// Check that we recursively lower nested ifs.
// CHECK-LABEL: func @conditional_nested(
func.func @conditional_nested(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>

  %cmp1 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: scf.if
  %if1 = "mhlo.if"(%cmp1) ({
    %cmp2 = "mhlo.compare"(%arg1, %cst) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %log = mhlo.log(%arg0) : (tensor<f32>) -> tensor<f32>

    // CHECK: scf.if
    %if2 = "mhlo.if"(%cmp2) ({
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    },  {
      "mhlo.return"(%log) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    "mhlo.return"(%if2) : (tensor<f32>) -> ()
  },  {
    %exp = mhlo.exponential(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%exp) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>

  func.return %if1 : tensor<f32>
}

// Test the two branches case as the common. Following tests verify degenerate
// behavior.
// CHECK-LABEL: func @case2(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case2(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {

  // CHECK-NEXT: %[[VAL_3:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[VAL_4:.*]] = "mhlo.compare"(%[[VAL_0]], %[[VAL_3]]) {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: %[[VAL_5:.*]] = tensor.extract %[[VAL_4]][] : tensor<i1>
  // CHECK: %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (tensor<4xf32>) {
  %1 = "mhlo.case"(%arg0) ({
      // CHECK: %[[VAL_7:.*]] = mhlo.log %[[VAL_1]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_7]] : tensor<4xf32>
      %2 = mhlo.log(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  }, {
      // CHECK: %[[VAL_8:.*]] = mhlo.exponential %[[VAL_2]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_8]] : tensor<4xf32>
      %3 = mhlo.exponential(%arg2) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>

  // CHECK: return %[[VAL_9:.*]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}


// CHECK-LABEL: func @case3(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:[0-9a-zA-Z]*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_3:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case3(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>, %arg3 : tensor<4xf32>) -> tensor<4xf32> {

  // CHECK-NEXT: %[[VAL_4:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_0]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: %[[VAL_6:.*]] = tensor.extract %[[VAL_5]][] : tensor<i1>
  // CHECK: %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (tensor<4xf32>) {
  %1 = "mhlo.case"(%arg0) ({
      // CHECK: %[[VAL_8:.*]] = mhlo.log %[[VAL_1]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_8]] : tensor<4xf32>
      %2 = mhlo.log(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  // CHECK-NEXT:   %[[VAL_9:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK:   %[[VAL_10:.*]] = "mhlo.compare"(%[[VAL_0]], %[[VAL_9]]) {compare_type = #mhlo<"comparison_type NOTYPE">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK:   %[[VAL_11:.*]] = tensor.extract %[[VAL_10]][] : tensor<i1>
  // CHECK:   %[[VAL_12:.*]] = scf.if %[[VAL_11]] -> (tensor<4xf32>) {
  }, {
      // CHECK: %[[VAL_13:.*]] = mhlo.exponential %[[VAL_2]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_13]] : tensor<4xf32>
      %3 = mhlo.exponential(%arg2) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  }, {

      // CHECK: %[[VAL_14:.*]] = mhlo.floor %[[VAL_3]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_14]] : tensor<4xf32>
      %3 = mhlo.floor(%arg3) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK:   scf.yield %[[VAL_15:.*]] : tensor<4xf32>

  // CHECK: return %[[VAL_16:.*]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// Case with only one branch is inlined rather than lowering.
// CHECK-LABEL: func @case0(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case0(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = "mhlo.case"(%arg0) ({
      // CHECK: %[[VAL_2:.*]] = mhlo.log %[[VAL_1]] : tensor<4xf32>
      %2 = mhlo.log(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[VAL_2]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// Case with only one branch is inlined. Check that we recursively lower.
// CHECK-LABEL: func @case0_nested(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case0_nested(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = "mhlo.case"(%arg0) ({
    %2 = "mhlo.case"(%arg0) ({
      // CHECK: %[[VAL_2:.*]] = mhlo.log %[[VAL_1]] : tensor<4xf32>
      %3 = mhlo.log(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%3) : (tensor<4xf32>) -> ()
    }) : (tensor<i32>) -> tensor<4xf32>
    "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[VAL_2]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

func.func @sort(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
  func.return %result#0, %result#1 : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL:   func @sort(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<2xi32>,
// CHECK-SAME:               %[[ARG1:.*]]: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
// Iterate through dimension 0
// CHECK-DAG:       %[[C0_0:.*]] = arith.constant 0
// CHECK-DAG:       %[[C0_1:.*]] = arith.constant 0
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[ARG0]], %[[C0_1]] : tensor<2xi32>
// CHECK-DAG:       %[[C1_0:.*]] = arith.constant 1 : index
// Iterate through dimension 1
// CHECK:           %[[VAL_6:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[C0_0]] to %[[VAL_4]] step %[[C1_0]] iter_args(%[[VAL_8:.*]] = %[[ARG0]], %[[VAL_9:.*]] = %[[ARG1]]) -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK-DAG:         %[[C0_2:.*]] = arith.constant 0
// CHECK-DAG:         %[[C1_1:.*]] = arith.constant 1
// CHECK-DAG:         %[[C2_0:.*]] = arith.constant 2
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[C2_0]], %[[C1_1]] : index
// Iterate through sorted dimension
// CHECK:             %[[VAL_14:.*]]:2 = scf.for %[[VAL_15:.*]] = %[[C0_2]] to %[[VAL_13]] step %[[C1_1]] iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_9]]) -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Extract each value twice because we are comparing both directions and haven't run CSE yet
// CHECK:               %[[VAL_19:.*]] = tensor.extract %[[VAL_8]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:               %[[VAL_20:.*]] = tensor.from_elements %[[VAL_19]] : tensor<i32>
// CHECK:               %[[VAL_21:.*]] = tensor.extract %[[VAL_8]]{{\[}}%[[VAL_18]]] : tensor<2xi32>
// CHECK:               %[[VAL_22:.*]] = tensor.from_elements %[[VAL_21]] : tensor<i32>
// CHECK:               %[[VAL_23:.*]] = tensor.extract %[[VAL_9]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:               %[[VAL_24:.*]] = tensor.from_elements %[[VAL_23]] : tensor<i32>
// CHECK:               %[[VAL_25:.*]] = tensor.extract %[[VAL_9]]{{\[}}%[[VAL_18]]] : tensor<2xi32>
// CHECK:               %[[VAL_26:.*]] = tensor.from_elements %[[VAL_25]] : tensor<i32>
// CHECK:               %[[VAL_27:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_28:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_30:.*]] = tensor.from_elements %[[VAL_29]] : tensor<i1>
// CHECK:               %[[VAL_31:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_32:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = tensor.from_elements %[[VAL_33]] : tensor<i1>
// Extract comparison results that were packed back into tensors by mhlo
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_34]][] : tensor<i1>
// CHECK:               %[[VAL_36:.*]] = tensor.extract %[[VAL_30]][] : tensor<i1>
// Determine if swapping should occur which happens only if NOT(CMP(A,B))  && CMP(B,A)
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[VAL_38:.*]] = arith.xori %[[VAL_35]], %[[TRUE]] : i1
// CHECK:               %[[VAL_39:.*]] = arith.andi %[[VAL_38]], %[[VAL_36]] : i1
// CHECK:               %[[VAL_40:.*]]:2 = scf.if %[[VAL_39]] -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Swap first pair of values
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:                 %[[VAL_43:.*]] = tensor.insert %[[VAL_42]] into %[[VAL_8]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:                 %[[VAL_45:.*]] = tensor.insert %[[VAL_44]] into %[[VAL_43]]{{\[}}%[[VAL_41]]] : tensor<2xi32>
// Swap second pair of values
// CHECK:                 %[[VAL_46:.*]] = tensor.extract %[[VAL_26]][] : tensor<i32>
// CHECK:                 %[[VAL_47:.*]] = tensor.insert %[[VAL_46]] into %[[VAL_9]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:                 %[[VAL_48:.*]] = tensor.extract %[[VAL_24]][] : tensor<i32>
// CHECK:                 %[[VAL_49:.*]] = tensor.insert %[[VAL_48]] into %[[VAL_47]]{{\[}}%[[VAL_41]]] : tensor<2xi32>
// CHECK:                 scf.yield %[[VAL_45]], %[[VAL_49]] : tensor<2xi32>, tensor<2xi32>
// CHECK:               } else {
// Don't swap
// CHECK:                 scf.yield %[[VAL_16]], %[[VAL_17]] : tensor<2xi32>, tensor<2xi32>
// CHECK:               }
// Propagate values back through the loops
// CHECK:               scf.yield %[[VAL_40:.*]]#0, %[[VAL_40]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:         }

func.func @dyn_sort(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  func.return %result#0, %result#1 : tensor<?xi32>, tensor<?xi32>
}
// CHECK-LABEL:   func @dyn_sort(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<?xi32>,
// CHECK-SAME:               %[[ARG1:.*]]: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
// Iterate through dimension 0
// CHECK-DAG:       %[[C0_0:.*]] = arith.constant 0
// CHECK-DAG:       %[[C0_1:.*]] = arith.constant 0
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[ARG0]], %[[C0_1]] : tensor<?xi32>
// CHECK-DAG:       %[[C1_0:.*]] = arith.constant 1 : index
// Iterate through dimension 1
// CHECK:           %[[VAL_6:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[C0_0]] to %[[VAL_4]] step %[[C1_0]] iter_args(%[[VAL_8:.*]] = %[[ARG0]], %[[VAL_9:.*]] = %[[ARG1]]) -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK-DAG:         %[[C0_2:.*]] = arith.constant 0
// CHECK-DAG:         %[[C1_1:.*]] = arith.constant 1
// CHECK-DAG:         %[[C0_3:.*]] = arith.constant 0
// CHECK:             %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0_3]] : tensor<?xi32>
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[DIM]], %[[C1_1]] : index
// Iterate through sorted dimension
// CHECK:             %[[VAL_14:.*]]:2 = scf.for %[[VAL_15:.*]] = %[[C0_2]] to %[[VAL_13]] step %[[C1_1]] iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_9]]) -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Extract each value twice because we are comparing both directions and haven't run CSE yet
// CHECK:               %[[VAL_19:.*]] = tensor.extract %[[VAL_8]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:               %[[VAL_20:.*]] = tensor.from_elements %[[VAL_19]] : tensor<i32>
// CHECK:               %[[VAL_21:.*]] = tensor.extract %[[VAL_8]]{{\[}}%[[VAL_18]]] : tensor<?xi32>
// CHECK:               %[[VAL_22:.*]] = tensor.from_elements %[[VAL_21]] : tensor<i32>
// CHECK:               %[[VAL_23:.*]] = tensor.extract %[[VAL_9]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:               %[[VAL_24:.*]] = tensor.from_elements %[[VAL_23]] : tensor<i32>
// CHECK:               %[[VAL_25:.*]] = tensor.extract %[[VAL_9]]{{\[}}%[[VAL_18]]] : tensor<?xi32>
// CHECK:               %[[VAL_26:.*]] = tensor.from_elements %[[VAL_25]] : tensor<i32>
// CHECK:               %[[VAL_27:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_28:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_30:.*]] = tensor.from_elements %[[VAL_29]] : tensor<i1>
// CHECK:               %[[VAL_31:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_32:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = tensor.from_elements %[[VAL_33]] : tensor<i1>
// Extract comparison results that were packed back into tensors by mhlo
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_34]][] : tensor<i1>
// CHECK:               %[[VAL_36:.*]] = tensor.extract %[[VAL_30]][] : tensor<i1>
// Determine if swapping should occur which happens only if NOT(CMP(A,B))  && CMP(B,A)
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[VAL_38:.*]] = arith.xori %[[VAL_35]], %[[TRUE]] : i1
// CHECK:               %[[VAL_39:.*]] = arith.andi %[[VAL_38]], %[[VAL_36]] : i1
// CHECK:               %[[VAL_40:.*]]:2 = scf.if %[[VAL_39]] -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Swap first pair of values
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:                 %[[VAL_43:.*]] = tensor.insert %[[VAL_42]] into %[[VAL_8]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:                 %[[VAL_45:.*]] = tensor.insert %[[VAL_44]] into %[[VAL_43]]{{\[}}%[[VAL_41]]] : tensor<?xi32>
// Swap second pair of values
// CHECK:                 %[[VAL_46:.*]] = tensor.extract %[[VAL_26]][] : tensor<i32>
// CHECK:                 %[[VAL_47:.*]] = tensor.insert %[[VAL_46]] into %[[VAL_9]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:                 %[[VAL_48:.*]] = tensor.extract %[[VAL_24]][] : tensor<i32>
// CHECK:                 %[[VAL_49:.*]] = tensor.insert %[[VAL_48]] into %[[VAL_47]]{{\[}}%[[VAL_41]]] : tensor<?xi32>
// CHECK:                 scf.yield %[[VAL_45]], %[[VAL_49]] : tensor<?xi32>, tensor<?xi32>
// CHECK:               } else {
// Don't swap
// CHECK:                 scf.yield %[[VAL_16]], %[[VAL_17]] : tensor<?xi32>, tensor<?xi32>
// CHECK:               }
// Propagate values back through the loops
// CHECK:               scf.yield %[[VAL_40:.*]]#0, %[[VAL_40]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:         }
