// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FILECHECK_OPTS="" FileCheck %s

func @main() -> tensor<f32> {
  %cst = constant dense<1> : tensor<i32>
  %cst_0 = constant dense<5.600000e+01> : tensor<f32>
  %cst_1 = constant dense<1.200000e+01> : tensor<f32>
  %cst_2 = constant dense<1.300000e+01> : tensor<f32>
  %0 = "mhlo.case"(%cst, %cst_0, %cst_1, %cst_2) ( {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
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

// CHECK: %[[INDEX:.*]] = s32[] constant(1)
// CHECK: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: ROOT %[[RESULT:.*]] = f32[] conditional(s32[] %[[INDEX]], f32[] %[[OPERAND_1]], f32[] %[[OPERAND_2]], f32[] %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}

// -----

func @main() -> (tensor<f32>, tensor<f32>) {
  %cst = constant dense<1> : tensor<i32>
  %cst_0 = constant dense<5.600000e+01> : tensor<f32>
  %cst_1 = constant dense<1.200000e+01> : tensor<f32>
  %cst_2 = constant dense<1.300000e+01> : tensor<f32>
  %0:2 = "mhlo.case"(%cst, %cst_0, %cst_1, %cst_2) ( {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
  ^bb0(%arg0: tensor<f32>):
    %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %0#0, %0#1 : tensor<f32>, tensor<f32>
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

// CHECK: %[[INDEX:.*]] = s32[] constant(1)
// CHECK: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: %[[TUPLE:.*]] = (f32[], f32[]) conditional(s32[] %[[INDEX]], f32[] %[[OPERAND_1]], f32[] %[[OPERAND_2]], f32[] %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}
// CHECK: %[[RES_1:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[TUPLE]]), index=0
// CHECK: %[[RES_2:.*]] = f32[] get-tuple-element((f32[], f32[]) %[[TUPLE]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[], f32[]) tuple(f32[] %[[RES_1]], f32[] %[[RES_2]])
