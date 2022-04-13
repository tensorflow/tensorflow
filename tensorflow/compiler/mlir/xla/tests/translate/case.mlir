// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FILECHECK_OPTS="" FileCheck %s

func.func @main() -> tensor<f32> {
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<5.600000e+01> : tensor<f32>
  %cst_1 = arith.constant dense<1.200000e+01> : tensor<f32>
  %cst_2 = arith.constant dense<1.300000e+01> : tensor<f32>
  %0 = "mhlo.case"(%cst) ({
    %1 = "mhlo.negate"(%cst_0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
    %1 = "mhlo.copy"(%cst_1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
    %1 = "mhlo.floor"(%cst_2) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK: %[[NEGATE_BRANCH:.*]] ({{.*}}: f32[]) -> f32[] {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   ROOT %[[RESULT:.*]] = f32[] negate(f32[] %[[ARG]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: f32[]) -> f32[] {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   ROOT %[[RESULT:.*]] = f32[] copy(f32[] %[[ARG]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: f32[]) -> f32[] {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   ROOT %[[RESULT:.*]] = f32[] floor(f32[] %[[ARG]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> f32[]

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: ROOT %[[RESULT:.*]] = f32[] conditional(s32[] %[[INDEX]], f32[] %[[OPERAND_1]], f32[] %[[OPERAND_2]], f32[] %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}

// -----

func.func @main() -> (tensor<f32>, tensor<f32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<5.600000e+01> : tensor<f32>
  %cst_1 = arith.constant dense<1.200000e+01> : tensor<f32>
  %cst_2 = arith.constant dense<1.300000e+01> : tensor<f32>
  %0:2 = "mhlo.case"(%cst) ({
    %1 = "mhlo.negate"(%cst_0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %1 = "mhlo.copy"(%cst_1) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %1 = "mhlo.floor"(%cst_2) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// CHECK: %[[NEGATE_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[NEGATE:.*]] = f32[] negate(f32[] %[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[NEGATE]], f32[] %[[NEGATE]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[COPY:.*]] = f32[] copy(f32[] %[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[COPY]], f32[] %[[COPY]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[FLOOR:.*]] = f32[] floor(f32[] %[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[FLOOR]], f32[] %[[FLOOR]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> (f32[], f32[])

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: %[[TUPLE:.*]] = (f32[], f32[]) conditional(s32[] %[[INDEX]], f32[] %[[OPERAND_1]], f32[] %[[OPERAND_2]], f32[] %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}
// CHECK: %[[RES_1:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[TUPLE]]), index=0
// CHECK: %[[RES_2:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[TUPLE]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[], f32[]) tuple(f32[] %[[RES_1]], f32[] %[[RES_2]])

// -----
// Test export mhlo::CaseOp with diffrent number of block-arguments (even 0).

func.func @main() -> (tensor<f32>, tensor<f32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %cst_0 = arith.constant dense<5.600000e+01> : tensor<f32>
  %cst_1 = arith.constant dense<1.200000e+01> : tensor<f32>
  %cst_2 = arith.constant dense<1.300000e+01> : tensor<f32>
  %0:2 = "mhlo.case"(%cst) ({
    %1 = "mhlo.negate"(%cst_0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %1 = "mhlo.copy"(%cst_1) : (tensor<f32>) -> tensor<f32>
    %2 = "mhlo.copy"(%cst_2) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    %cst_3 = arith.constant dense<1.300000e+01> : tensor<f32>
    %1 = "mhlo.floor"(%cst_3) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// CHECK: %[[NEGATE_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[NEGATE:.*]] = f32[] negate(f32[] %[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[NEGATE]], f32[] %[[NEGATE]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = (f32[], f32[]) parameter(0)
// CHECK-DAG:   %[[GTE1:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[ARG]]), index=0
// CHECK-DAG:   %[[COPY1:.*]] = f32[] copy(f32[] %[[GTE1]])
// CHECK-DAG:   %[[GTE2:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[ARG]]), index=1
// CHECK-DAG:   %[[COPY2:.*]] = f32[] copy(f32[] %[[GTE2]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[COPY1]], f32[] %[[COPY2]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: ()) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = () parameter(0)
// CHECK:   %[[CST:.*]] = f32[] constant
// CHECK:   %[[FLOOR:.*]] = f32[] floor(f32[] %[[CST]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(f32[] %[[FLOOR]], f32[] %[[FLOOR]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> (f32[], f32[])

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK-DAG: %[[TUPLE1:.*]] = (f32[], f32[]) tuple(f32[] %[[OPERAND_2]], f32[] %[[OPERAND_3]])
// CHECK-DAG: %[[TUPLE2:.*]] = () tuple()

// CHECK: %[[COND:.*]] = (f32[], f32[]) conditional(s32[] %[[INDEX]], f32[] %[[OPERAND_1]], (f32[], f32[]) %[[TUPLE1]], () %[[TUPLE2]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}

// CHECK: %[[RES_1:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[COND]]), index=0
// CHECK: %[[RES_2:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[COND]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[], f32[]) tuple(f32[] %[[RES_1]], f32[] %[[RES_2]])
