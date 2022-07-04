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