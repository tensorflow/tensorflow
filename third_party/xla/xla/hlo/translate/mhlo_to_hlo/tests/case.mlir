// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FILECHECK_OPTS="" FileCheck %s

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
// CHECK:   ROOT %[[RESULT:.*]] = f32[] negate(%[[ARG]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: f32[]) -> f32[] {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   ROOT %[[RESULT:.*]] = f32[] copy(%[[ARG]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: f32[]) -> f32[] {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   ROOT %[[RESULT:.*]] = f32[] floor(%[[ARG]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> f32[]

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: ROOT %[[RESULT:.*]] = f32[] conditional(%[[INDEX]], %[[OPERAND_1]], %[[OPERAND_2]], %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}

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
// CHECK:   %[[NEGATE:.*]] = f32[] negate(%[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[NEGATE]], %[[NEGATE]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[COPY:.*]] = f32[] copy(%[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[COPY]], %[[COPY]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: f32[]) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = f32[] parameter(0)
// CHECK:   %[[FLOOR:.*]] = f32[] floor(%[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[FLOOR]], %[[FLOOR]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> (f32[], f32[])

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK: %[[TUPLE:.*]] = (f32[], f32[]) conditional(%[[INDEX]], %[[OPERAND_1]], %[[OPERAND_2]], %[[OPERAND_3]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}
// CHECK: %[[RES_1:.*]] = f32[] get-tuple-element(%[[TUPLE]]), index=0
// CHECK: %[[RES_2:.*]] = f32[] get-tuple-element(%[[TUPLE]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[], f32[]) tuple(%[[RES_1]], %[[RES_2]])

// -----
// Test export mhlo::CaseOp with different number of block-arguments (even 0).

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
// CHECK:   %[[NEGATE:.*]] = f32[] negate(%[[ARG]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[NEGATE]], %[[NEGATE]])
// CHECK: }

// CHECK: %[[COPY_BRANCH:.*]] ({{.*}}: (f32[], f32[])) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = (f32[], f32[]) parameter(0)
// CHECK-DAG:   %[[GTE1:.*]] = f32[] get-tuple-element(%[[ARG]]), index=0
// CHECK-DAG:   %[[COPY1:.*]] = f32[] copy(%[[GTE1]])
// CHECK-DAG:   %[[GTE2:.*]] = f32[] get-tuple-element(%[[ARG]]), index=1
// CHECK-DAG:   %[[COPY2:.*]] = f32[] copy(%[[GTE2]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[COPY1]], %[[COPY2]])
// CHECK: }

// CHECK: %[[FLOOR_BRANCH:.*]] ({{.*}}: ()) -> (f32[], f32[]) {
// CHECK:   %[[ARG:.*]] = () parameter(0)
// CHECK:   %[[CST:.*]] = f32[] constant
// CHECK:   %[[FLOOR:.*]] = f32[] floor(%[[CST]])
// CHECK:   ROOT %[[TUPLE:.*]] = (f32[], f32[]) tuple(%[[FLOOR]], %[[FLOOR]])
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME:  () -> (f32[], f32[])

// CHECK-DAG: %[[INDEX:.*]] = s32[] constant(1)
// CHECK-DAG: %[[OPERAND_1:.*]] = f32[] constant(56)
// CHECK-DAG: %[[OPERAND_2:.*]] = f32[] constant(12)
// CHECK-DAG: %[[OPERAND_3:.*]] = f32[] constant(13)
// CHECK-DAG: %[[TUPLE1:.*]] = (f32[], f32[]) tuple(%[[OPERAND_2]], %[[OPERAND_3]])
// CHECK-DAG: %[[TUPLE2:.*]] = () tuple()

// CHECK: %[[COND:.*]] = (f32[], f32[]) conditional(%[[INDEX]], %[[OPERAND_1]], %[[TUPLE1]], %[[TUPLE2]]), branch_computations={%[[NEGATE_BRANCH]], %[[COPY_BRANCH]], %[[FLOOR_BRANCH]]}

// CHECK: %[[RES_1:.*]] = f32[] get-tuple-element(%[[COND]]), index=0
// CHECK: %[[RES_2:.*]] = f32[] get-tuple-element(%[[COND]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[], f32[]) tuple(%[[RES_1]], %[[RES_2]])


// -----
// Test export mhlo::CaseOp with captured variables that have shardings.


func.func @main(%arg0: tensor<i32>, %arg1: tensor<8xi64> {mhlo.sharding = "{manual}"}, %arg3: tensor<i1>) {
  %0 = mhlo.custom_call @SPMDFullToShardShape(%arg0) {mhlo.sharding = "{manual}"} : (tensor<i32>) -> tensor<i32>
  %1 = mhlo.convert %arg3 {mhlo.sharding = "{manual}"} : (tensor<i1>) -> tensor<i32>
  "mhlo.case"(%1) ({
    mhlo.return
  }, {
    mhlo.custom_call @xla_ffi_python_cpu_callback(%0, %arg1) {mhlo.sharding = "{manual}"} : (tensor<i32>, tensor<8xi64>) -> ()
    mhlo.return
  }) {mhlo.sharding = "{replicated}"} : (tensor<i32>) -> ()
  return
}

// CHECK: %[[EMPTY_BRANCH:.*]] ({{.*}}: ()) -> () {
// CHECK:   %[[ARG:.*]] = () parameter(0)
// CHECK:   ROOT %{{.*}} = () tuple()
// CHECK: }

// CHECK: %[[CALLBACK_BRANCH:.*]] ({{.*}}: (s32[], s64[8])) -> () {
// CHECK:   %[[ARG:.*]] = (s32[], s64[8]) parameter(0)
// CHECK:   %[[ELEMENT_0:.*]] = s32[] get-tuple-element(%[[ARG]]), index=0, sharding={manual}
// CHECK:   %[[ELEMENT_1:.*]] = s64[8] get-tuple-element(%[[ARG]]), index=1, sharding={manual}
// CHECK:   %{{.*}} = () custom-call(%[[ELEMENT_0]], %[[ELEMENT_1]]),
// CHECK-SAME{LITERAL}: custom_call_target="xla_ffi_python_cpu_callback", sharding={{manual}}
// CHECK:   ROOT %{{.*}} = () tuple()
// CHECK: }

// CHECK-LABEL: ENTRY
// CHECK-SAME: ({{.*}}: s32[], {{.*}}: s64[8], {{.*}}: pred[]) -> () {
// CHECK-DAG: %[[ARG_2:.*]] = pred[] parameter(2)
// CHECK-DAG: %[[CONVERT:.*]] = s32[] convert(%[[ARG_2]]), sharding={manual}
// CHECK-DAG: %[[EMPTY_TUPLE:.*]] = () tuple()
// CHECK-DAG: %[[ARG_0:.*]] = s32[] parameter(0)
// CHECK-DAG: %[[FULL_TO_SHARD:.*]] = s32[] custom-call(%[[ARG_0]]), custom_call_target="SPMDFullToShardShape", sharding={manual}
// CHECK-DAG: %[[ARG_1:.*]] = s64[8] parameter(1), sharding={manual}
// CHECK-DAG: %[[ARG_TUPLE:.*]] = (s32[], s64[8]) tuple(%[[FULL_TO_SHARD]], %[[ARG_1]]),
// CHECK-SAME{LITERAL}: sharding={{manual}, {manual}}
// CHECK-DAG: %{{.*}} = () conditional(%[[CONVERT]], %[[EMPTY_TUPLE]], %[[ARG_TUPLE]]), branch_computations={%[[EMPTY_BRANCH]], %[[CALLBACK_BRANCH]]},
// CHECK-SAME{LITERAL}: sharding={{replicated}}
// CHECK-DAG: ROOT %{{.*}} = () tuple()
