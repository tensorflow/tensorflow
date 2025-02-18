// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: HloModule main
// CHECK: [[R0:%.+]] ([[A0:.+]]: f32[]) -> f32[] {
// CHECK:   %[[A0]] = f32[] parameter(0)

// CHECK: [[R1:%.+]] ([[A0:.+]]: f32[]) -> f32[] {
// CHECK:   %[[A0]] = f32[] parameter(0)

// CHECK: ENTRY
// CHECK-NEXT:   %[[A0:.+]] = f32[] parameter(0)
func.func @main(%arg0: tensor<f32>) -> tuple<tensor<f32>> {
  // CHECK:   %[[VAL0:.+]] = f32[] constant(10)
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  // CHECK:   %[[VAL1:.+]] = pred[] compare(f32[] %[[A0]], f32[] %[[VAL0]]), direction=LT
  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK:   %[[VAL2:.+]] = f32[] conditional(pred[] %[[VAL1]], f32[] %[[A0]], f32[] %[[A0]]), true_computation=[[R0]], false_computation=[[R1]]
  %2 = "mhlo.if"(%0) ({
    %6 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%6) : (tensor<f32>) -> ()
  },  {
    %6 = "mhlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%6) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:   ROOT %[[VAL3:.+]] = (f32[]) tuple(f32[] %[[VAL2]])
  %3 = "mhlo.tuple"(%2) : (tensor<f32>) -> tuple<tensor<f32>>
  func.return %3 : tuple<tensor<f32>>
}

// -----
// Test export mhlo::IfOp with multiple args, but same numbers of args for the
// branches.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %2:2 = "mhlo.if"(%0) ({
    %log = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
    %add = mhlo.add %log, %arg1 : tensor<f32>
    "mhlo.return"(%add, %log) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    "mhlo.return"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %2#0 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0:.+]]: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = (f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK: }

// CHECK: [[R1:%.+]] ([[A0:.+]]: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = (f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK: }

// CHECK: ENTRY
// CHECK-DAG: %[[A0:.+]] = f32[] parameter(0)
// CHECK-DAG: %[[A1:.+]] = f32[] parameter(1)
// CHECK-DAG: %[[TUPLE1:.+]] = (f32[], f32[]) tuple(f32[] %[[A0]], f32[] %[[A1]])
// CHECK-DAG: %[[TUPLE2:.+]] = (f32[], f32[]) tuple(f32[] %[[A0]], f32[] %[[A1]])
// CHECK-DAG: %[[COND:.+]] = (f32[], f32[]) conditional(pred[] %[[PRED:.+]], (f32[], f32[]) %[[TUPLE1]], (f32[], f32[]) %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]

// -----
// Test export mhlo::IfOp with multiple args, but different numbers of args for
// branches.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %2:2 = "mhlo.if"(%0) ({
    %log = "mhlo.log"(%cst) : (tensor<f32>) -> tensor<f32>
    %add = mhlo.add %log, %arg1 : tensor<f32>
    "mhlo.return"(%arg0, %add) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    "mhlo.return"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %2#0 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0:.+]]: (f32[], f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   (f32[], f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: [[R1:%.+]] ([[A0:.+]]: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = (f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: ENTRY
// CHECK-DAG: %[[A0:.+]] = f32[] parameter(0)
// CHECK-DAG: %[[CST:.+]] = f32[] constant(10)
// CHECK-DAG: %[[A1:.+]] = f32[] parameter(1)
// CHECK-DAG: %[[TUPLE1:.+]] = (f32[], f32[], f32[]) tuple(f32[] %[[CST]], f32[] %[[A1]], f32[] %[[A0]])
// CHECK-DAG: %[[TUPLE2:.+]] = (f32[], f32[]) tuple(f32[] %[[A0]], f32[] %[[A1]])
// CHECK: %[[COND:.+]] = (f32[], f32[]) conditional(pred[] %[[PRED:.+]], (f32[], f32[], f32[]) %[[TUPLE1]], (f32[], f32[]) %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]

// -----
// Test export mhlo::IfOp with false branch having no implict captures.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %2:2 = "mhlo.if"(%0) ({
    %log = "mhlo.log"(%cst) : (tensor<f32>) -> tensor<f32>
    %add = mhlo.add %log, %arg1 : tensor<f32>
    "mhlo.return"(%arg0, %add) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %cst1 = arith.constant  dense<1.000000e+01> : tensor<f32>
    "mhlo.return"(%cst1, %cst1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %2#0 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0:.+]]: (f32[], f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   (f32[], f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: [[R1:%.+]] ([[A0:.+]]: ()) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = () parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: ENTRY
// CHECK-DAG: %[[A0:.+]] = f32[] parameter(0)
// CHECK-DAG: %[[CST:.+]] = f32[] constant(10)
// CHECK-DAG: %[[A1:.+]] = f32[] parameter(1)
// CHECK-DAG: %[[TUPLE1:.+]] = (f32[], f32[], f32[]) tuple(f32[] %[[CST]], f32[] %[[A1]], f32[] %[[A0]])
// CHECK-DAG: %[[TUPLE2:.+]] = () tuple()
// CHECK: %[[COND:.+]] = (f32[], f32[]) conditional(pred[] %[[PRED:.+]], (f32[], f32[], f32[]) %[[TUPLE1]], () %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]

// -----
// Test export mhlo::IfOp with true branch having no implict captures.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %2:2 = "mhlo.if"(%0) ({
    %cst1 = arith.constant  dense<1.000000e+01> : tensor<f32>
    "mhlo.return"(%cst1, %cst1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %log = "mhlo.log"(%cst) : (tensor<f32>) -> tensor<f32>
    %add = mhlo.add %log, %arg1 : tensor<f32>
    "mhlo.return"(%arg0, %add) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %2#0 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0:.+]]: ()) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = () parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: [[R1:%.+]] ([[A0:.+]]: (f32[], f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   (f32[], f32[], f32[]) parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: ENTRY
// CHECK-DAG: %[[A0:.+]] = f32[] parameter(0)
// CHECK-DAG: %[[CST:.+]] = f32[] constant(10)
// CHECK-DAG: %[[A1:.+]] = f32[] parameter(1)
// CHECK-DAG: %[[TUPLE1:.+]] = () tuple()
// CHECK-DAG: %[[TUPLE2:.+]] = (f32[], f32[], f32[]) tuple(f32[] %[[CST]], f32[] %[[A1]], f32[] %[[A0]])
// CHECK: %[[COND:.+]] = (f32[], f32[]) conditional(pred[] %[[PRED:.+]], () %[[TUPLE1]], (f32[], f32[], f32[]) %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]

// -----
// Test export mhlo::IfOp with both branches having no implict captures.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0 = "mhlo.compare"(%arg0, %cst) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  %2:2 = "mhlo.if"(%0) ({
    %cst1 = arith.constant  dense<1.000000e+01> : tensor<f32>
    "mhlo.return"(%cst1, %cst1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %cst2 = arith.constant  dense<1.000000e+01> : tensor<f32>
    "mhlo.return"(%cst2, %cst2) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %2#0 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0:.+]]: ()) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = () parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: [[R1:%.+]] ([[A0:.+]]: ()) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A0]] = () parameter(0)
// CHECK:  ROOT %[[TUPLE:.+]] = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: ENTRY
// CHECK: %[[TUPLE1:.+]] = () tuple()
// CHECK: %[[TUPLE2:.+]] = () tuple()
// CHECK: %[[COND:.+]] = (f32[], f32[]) conditional(pred[] %[[PRED:.+]], () %[[TUPLE1]], () %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]

// -----
// Test export nested mhlo::IfOp.
// outer-if: returns mutiple values
//   true-branch:
//     inner-if: returns single value
//        true-branch: captures 1 value defined in true-branch; corresponding
//                     xla parameter is of type non-tuple.
//        false-branch: no implicit captures
//     false-branch: Uses 2 implict captures from above; corresponding xla
//                   parameter is of type tuple.

func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant  dense<1.000000e+01> : tensor<f32>

  %0:2 = "mhlo.if"(%arg0) ({
        // R2
    %1 = mhlo.constant dense<false> : tensor<i1>
    %cst0 = mhlo.constant dense<1.000000e+01> : tensor<f32>

    %2 = "mhlo.if"(%1) ({
        // R0
      "mhlo.return"(%cst0) : (tensor<f32>) -> ()
    },  {
        // R1
      %cst1 = mhlo.constant dense<1.000000e+01> : tensor<f32>
      "mhlo.return"(%cst1) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>

    "mhlo.return"(%2, %2) : (tensor<f32>, tensor<f32>) -> ()

  },  {

        // R3
    "mhlo.return"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  func.return %0#1 : tensor<f32>
}

// CHECK-LABEL: HloModule main

// CHECK: [[R0:%.+]] ([[A0_NON_TUPLE:.+]]: f32[]) -> f32[] {
// CHECK-NEXT:   ROOT %[[A0_NON_TUPLE]] = f32[] parameter(0)

// CHECK: [[R1:%.+]] ([[A1_EMPTY_TUPLE:.+]]: ()) -> f32[] {
// CHECK-NEXT:  %[[A1_EMPTY_TUPLE]] = () parameter(0)
// CHECK-NEXT:  ROOT %[[CST1:.+]] = f32[] constant(10)
// CHECK-NEXT: }

// CHECK: [[R2:%.+]] ([[A2_EMPTY_TUPLE:.+]]: ()) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A2_EMPTY_TUPLE]] = () parameter(0)
// CHECK-DAG:   %[[CST2:.+]] = f32[] constant(10)
// CHECK-DAG: %[[TUPLE2:.+]] = () tuple()
// CHECK:  %[[COND2:.+]] = f32[] conditional(pred[] %{{.+}}, f32[] %[[CST2]], () %[[TUPLE2]]), true_computation=[[R0]], false_computation=[[R1]]
// CHECK:  ROOT %tuple.{{[0-9]+}} = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: [[R3:%.+]] ([[A3_TUPLE:.+]]: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK-NEXT:   %[[A3_TUPLE]] = (f32[], f32[]) parameter(0)
// CHECK:  ROOT %{{.+}} = (f32[], f32[]) tuple
// CHECK-NEXT: }

// CHECK: ENTRY
// CHECK-DAG: %[[CST:.+]] = f32[] constant(10)
// CHECK-DAG: %[[A0:.+]] = pred[] parameter(0)
// CHECK-DAG: %[[A1:.+]] = f32[] parameter(1)
// CHECK-DAG: %[[A2:.+]] = f32[] parameter(2)
// CHECK-DAG: %[[TUPLE1:.+]]  = () tuple()
// CHECK-DAG: %[[TUPLE2:.+]]  = (f32[], f32[]) tuple(f32[] %[[A1]], f32[] %[[A2]])
// CHECK: (f32[], f32[]) conditional(pred[] %[[A0]], () %[[TUPLE1]], (f32[], f32[]) %[[TUPLE2]]), true_computation=[[R2]], false_computation=[[R3]]
