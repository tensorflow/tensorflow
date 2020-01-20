/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_irgen_test_base.h"

namespace xla {
namespace mlir_gpu {

class LhloGenTest : public MlirIrGenTestBase {};

TEST_F(LhloGenTest, Const) {
  CompileAndVerifyIr(R"(
HloModule Const

ENTRY %Const () -> s32[100] {
  %const.0 = s32[] constant(10)
  ROOT %broadcast.0 = s32[100]{0} broadcast(s32[] %const.0), dimensions={}
})",
                     R"(
;CHECK: func @constant(%[[ARG0:.*]]: memref<i32>)
;CHECK:   "xla_lhlo.constant"(%[[ARG0]]) {value = dense<10> : tensor<i32>}
;CHECK: func @broadcast(%[[ARG1:.*]]: memref<i32>, %[[ARG2:.*]]: memref<100xi32>)
;CHECK:   "xla_lhlo.broadcast_in_dim"(%[[ARG1]], %[[ARG2]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>}
)",
                     LoweringStage::LHLO);
}

TEST_F(LhloGenTest, BrokenAdd) {
  CompileAndVerifyErrors(
      R"(
HloModule Add

ENTRY %Add (x: f32[2,2,2], y: f32[2,2,2]) -> f32[2,2,2] {
  %x = f32[2,2,2]{2,1,0} parameter(0)
  %y = f32[2,2,2]{2,1,0} parameter(1)
  ROOT %add = f32[2,2,2]{2,1,0} add(f32[2,2,2]{2,1,0} %x, f32[2,2,2]{2,1,0} %y)
})",
      R"(CHECK: ERRORS FOUND: [%add = f32[2,2,2]{2,1,0} add(f32[2,2,2]{2,1,0} %x, f32[2,2,2]{2,1,0} %y): failed for testing: xla_lhlo.add; failed for testing: std.return])",
      LoweringStage::LHLO);
}

TEST_F(LhloGenTest, Add) {
  CompileAndVerifyIr(R"(
HloModule Add

ENTRY %Add (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @add(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Compare) {
  CompileAndVerifyIr(R"(
HloModule Compare

ENTRY %Compare (x: f32[2,2], y: f32[2,2]) -> pred[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %compare = pred[2,2]{1,0} compare(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y), direction=EQ
})",
                     R"(
;CHECK: func @compare(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[PRED:.*]]: [[PRED_TYPE:.*]]) {
;CHECK:   "xla_lhlo.compare"(%[[ARG0]], %[[ARG1]], %[[PRED]])
;CHECK: {comparison_direction = "EQ"} : ([[TYPE]], [[TYPE]], [[PRED_TYPE]]) -> ()
;CHECK: }
)");
}

TEST_F(LhloGenTest, Copy) {
  CompileAndVerifyIr(R"(
HloModule Copy

ENTRY %Copy (x: f32[2,4]) -> f32[2,4] {
  %x = f32[2,4] parameter(0)
  ROOT %copy = f32[2,4] copy(f32[2,4] %x)
})",
                     R"(
;CHECK: func @copy(%[[OPERAND:.*]]: memref<2x4xf32>, %[[RESULT:.*]]: memref<2x4xf32>) {
;CHECK:   "xla_lhlo.copy"(%[[OPERAND]], %[[RESULT]]) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
      )");
}

TEST_F(LhloGenTest, Select) {
  CompileAndVerifyIr(R"(
HloModule Select

ENTRY %Select (p: pred[2,2], x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %p = pred[2,2]{1,0} parameter(0)
  %x = f32[2,2]{1,0} parameter(1)
  %y = f32[2,2]{1,0} parameter(2)
  ROOT %select = f32[2,2]{1,0} select(pred[2,2]{1,0} %p, f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @select(%[[PRED:.*]]: [[PRED_TYPE:.*]], %[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.select"(%[[PRED]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) : ([[PRED_TYPE]], [[TYPE]], [[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Exp) {
  CompileAndVerifyIr(R"(
HloModule Exp

ENTRY %Exp (x: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  ROOT %exp = f32[2,2]{1,0} exponential(f32[2,2]{1,0} %x)
})",
                     R"(
;CHECK: func @exponential(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.exp"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, AddInGPUDialect) {
  CompileAndVerifyIr(R"(
HloModule Add

ENTRY %Add (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @add(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]) {
;CHECK: "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ARG0]], %[[ARG1]], %[[ARG2]]
;CHECK: }
;CHECK: func @add_kernel(%[[ARG0]]: [[TYPE]], %[[ARG1]]: [[TYPE]], %[[ARG2]]: [[TYPE]]
;CHECK-DAG: std.subview %[[ARG0]]{{\[}}[[INDEX:.*]]]
;CHECK-DAG: std.subview %[[ARG1]]{{\[}}[[INDEX]]]
;CHECK-DAG: std.subview %[[ARG2]]{{\[}}[[INDEX]]]
;CHECK: %[[VAL1:.*]] = load %{{.*\[}}[[INDEX:.*]]]
;CHECK: %[[VAL2:.*]] = load %{{.*\[}}[[INDEX]]]
;CHECK: %[[RES:.*]] = addf %[[VAL1]], %[[VAL2]]
;CHECK: store %[[RES]], %{{.*\[}}[[INDEX]]]
      )",
                     LoweringStage::GPU);
}

// This test verifies that the kernel signature is amended correctly. The actual
// body of the generated function does not matter, it is already checked at the
// GPU level above.
TEST_F(LhloGenTest, AddAsKernel) {
  CompileAndVerifyIr(R"(
HloModule Add

ENTRY %Add (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @add_kernel(%[[ARG0:.*]]: [[TYPE:!llvm<.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]

;CHECK: %[[DESC0:.*]] = llvm.alloca %1 x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
;CHECK: %[[CAST0:.*]] = llvm.bitcast %[[ARG0]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP0P:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: llvm.store %[[CAST0]], %[[GEP0P]]
;CHECK: %[[CAST0:.*]] = llvm.bitcast %[[ARG0]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP0P:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: llvm.store %[[CAST0]], %[[GEP0P]]
;CHECK: %[[GEP0O:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: llvm.store %{{.*}}, %[[GEP0O]]
;CHECK: %[[GEP0S0:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: %[[CST0S0:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST0S0]], %[[GEP0S0]]
;CHECK: %[[GEP0S1:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: %[[CST0S1:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST0S1]], %[[GEP0S1]]
;CHECK: %[[GEP0ST0:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: llvm.store %{{.*}}, %[[GEP0ST0]]
;CHECK: %[[GEP0ST1:.*]] = llvm.getelementptr %[[DESC0]]
;CHECK: llvm.store %{{.*}}, %[[GEP0ST1]]

;CHECK: %[[DESC1:.*]] = llvm.alloca %1 x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
;CHECK: %[[CAST1:.*]] = llvm.bitcast %[[ARG1]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP1P:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: llvm.store %[[CAST1]], %[[GEP1P]]
;CHECK: %[[CAST1:.*]] = llvm.bitcast %[[ARG1]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP1P:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: llvm.store %[[CAST1]], %[[GEP1P]]
;CHECK: %[[GEP1O:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: llvm.store %{{.*}}, %[[GEP1O]]
;CHECK: %[[GEP1S0:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: %[[CST1S0:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST1S0]], %[[GEP1S0]]
;CHECK: %[[GEP1S1:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: %[[CST1S1:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST1S1]], %[[GEP1S1]]
;CHECK: %[[GEP1ST0:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: llvm.store %{{.*}}, %[[GEP1ST0]]
;CHECK: %[[GEP1ST1:.*]] = llvm.getelementptr %[[DESC1]]
;CHECK: llvm.store %{{.*}}, %[[GEP1ST1]]

;CHECK: %[[DESC2:.*]] = llvm.alloca %1 x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
;CHECK: %[[CAST2:.*]] = llvm.bitcast %[[ARG2]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP2P:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: llvm.store %[[CAST2]], %[[GEP2P]]
;CHECK: %[[CAST2:.*]] = llvm.bitcast %[[ARG2]] : [[TYPE]] to !llvm<"float*">
;CHECK: %[[GEP2P:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: llvm.store %[[CAST2]], %[[GEP2P]]
;CHECK: %[[GEP2O:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: llvm.store %{{.*}}, %[[GEP2O]]
;CHECK: %[[GEP2S0:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: %[[CST2S0:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST2S0]], %[[GEP2S0]]
;CHECK: %[[GEP2S1:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: %[[CST2S1:.*]] = llvm.mlir.constant(2 : i64) : !llvm.i64
;CHECK: llvm.store %[[CST2S1]], %[[GEP2S1]]
;CHECK: %[[GEP2ST0:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: llvm.store %{{.*}}, %[[GEP2ST0]]
;CHECK: %[[GEP2ST1:.*]] = llvm.getelementptr %[[DESC2]]
;CHECK: llvm.store %{{.*}}, %[[GEP2ST1]]
      )",
                     LoweringStage::KERNEL);
}

TEST_F(LhloGenTest, AddMultiply) {
  CompileAndVerifyIr(R"(
HloModule AddMultiply

ENTRY %AddMultiply (x: f32[2,2], y: f32[2,2], z: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  %z = f32[2,2]{1,0} parameter(2)
  %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
  ROOT %mul = f32[2,2]{1,0} multiply(f32[2,2]{1,0} %add, f32[2,2]{1,0} %z)
})",
                     R"(
;CHECK: func @fusion(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]], %[[RESULT:.*]]: [[TYPE]])
;CHECK: "xla_lhlo.fusion"() ( {
;CHECK:   %[[REF0:.*]] = tensor_load %[[ARG0]] : [[TYPE]]
;CHECK:   %[[REF1:.*]] = tensor_load %[[ARG1]] : [[TYPE]]
;CHECK:   %[[REF2:.*]] = tensor_load %[[ARG2]] : [[TYPE]]
;CHECK:   %[[ADD:.*]] = xla_hlo.add %[[REF1]], %[[REF2]] 
;CHECK:   %[[MUL:.*]] = xla_hlo.mul %[[ADD]], %[[REF0]]
;CHECK:   tensor_store %[[MUL]], %[[RESULT]]
;CHECK:   "xla_lhlo.terminator"()
;CHECK-NEXT: }
      )");
}

TEST_F(LhloGenTest, IotaAddMultiply) {
  CompileAndVerifyIr(R"(
HloModule AddMultiply

ENTRY %AddMultiply (x: s32[2,2], y: s32[2,2]) -> s32[2,2] {
  %x = s32[2,2]{1,0} parameter(0)
  %y = s32[2,2]{1,0} parameter(1)

  %add = s32[2,2]{1,0} add(s32[2,2]{1,0} %x, s32[2,2]{1,0} %y)
  %iota = s32[2, 2]{1,0} iota(), iota_dimension=0

  ROOT %mul = s32[2,2]{1,0} multiply(s32[2,2]{1,0} %add, s32[2,2]{1,0} %iota)
})",
                     R"(
;CHECK-NOT:  store
;CHECK:      %[[RESULT:.*]] = muli %{{.*}}, %{{.*}}
;CHECK:      store %[[RESULT]]
)",
                     LoweringStage::GPU);
}

TEST_F(LhloGenTest, AddMultiplyGPU) {
  CompileAndVerifyIr(R"(
HloModule AddMultiply

ENTRY %AddMultiply (x: f32[2,2], y: f32[2,2], z: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  %z = f32[2,2]{1,0} parameter(2)
  %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
  ROOT %mul = f32[2,2]{1,0} multiply(f32[2,2]{1,0} %add, f32[2,2]{1,0} %z)
})",
                     R"(
;CHECK: func @fusion_kernel(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]], %[[RESULT:.*]]: [[TYPE]])
;CHECK-DAG: std.subview %[[ARG0]]{{\[}}[[INDEX:.*]]]
;CHECK-DAG: std.subview %[[ARG1]]{{\[}}[[INDEX]]]
;CHECK-DAG: std.subview %[[ARG2]]{{\[}}[[INDEX]]]
;CHECK-DAG: std.subview %[[RESULT]]{{\[}}[[INDEX]]]
;CHECK:   %[[V0:.*]] = load %{{.*\[}}[[CSTIDX:.*]]]
;CHECK:   %[[V1:.*]] = load %{{.*\[}}[[CSTIDX:.*]]]
;CHECK:   %[[ADD:.*]] = addf %[[V0]], %[[V1]]
;CHECK:   %[[V2:.*]] = load %{{.*\[}}[[CSTIDX:.*]]]
;CHECK:   %[[MUL:.*]] = mulf %[[ADD]], %[[V2]]
;CHECK:   store %[[MUL]], %{{.*\[}}[[CSTIDX:.*]]]
;CHECK-NEXT: return
      )",
                     LoweringStage::GPU);
}

// TODO(b/137624192): Reenable once we can fuse reductions.
// TEST_F(LhloGenTest, FusedReduce) {
//   CompileAndVerifyIr(R"(
// HloModule FusedReduce
//
// %add (x: f32[], y: f32[]) -> f32[] {
//   %x = f32[] parameter(0)
//   %y = f32[] parameter(1)
//   ROOT %add = f32[] add(f32[] %x, f32[] %y)
// }
//
// %fused_computation (param: f32[100,10]) -> f32[10] {
//   %param = f32[100,10] parameter(0)
//   %constant = f32[] constant(0)
//   ROOT %reduce = f32[10]{0} reduce(f32[100,10]{1,0} %param, f32[] %constant),
//       dimensions={0}, to_apply=%add
// }
//
// ENTRY %FusedReduce (x: f32[100,10]) -> f32[10] {
//   %x = f32[100,10] parameter(0)
//   ROOT %fusion = f32[10]{0} fusion(f32[100,10]{1,0} %x), kind=kInput,
//       calls=%fused_computation
// }
// )",
//                      R"(
// ;CHECK: func @fusion(%[[ARG0:.*]]: [[TYPE:.*]], %[[RESULT:.*]]: [[RTYPE:.*]])
// ;CHECK: "xla_lhlo.fusion"() ( {
// ;CHECK:   %[[REF0:.*]] = tensor_load %arg0 : [[TYPE]]
// ;CHECK:   %[[CT0:.*]] = xla_hlo.constant dense<0.000000e+00>
// ;CHECK:   %[[RED:.*]] = "xla_hlo.reduce"(%0, %1) ( {
// ;CHECK:     ^bb0(%[[BARG0:.*]]: [[ETYPE:.*]], %[[BARG1:.*]]: [[ETYPE]])
// ;CHECK:       %[[ADD:.*]] = xla_hlo.add %[[BARG0]], %[[BARG1]] : [[ETYPE]]
// ;CHECK:       "xla_hlo.return"(%[[ADD]])
// ;CHECK:     })
// ;CHECK:   tensor_store %[[RED]], %[[RESULT]] : [[RTYPE]]
// ;CHECK:   "xla_lhlo.terminator"()
// ;CHECK-NEXT: })
//       )");
// }

TEST_F(LhloGenTest, Broadcast) {
  CompileAndVerifyIr(R"(
HloModule Broadcast

ENTRY %Broadcast (x: f32[10]) -> f32[10, 5] {
  %x = f32[10]{0} parameter(0)
  ROOT %broadcast = f32[10, 5]{1,0} broadcast(f32[10]{0} %x), dimensions={0}
})",
                     R"(
;CHECK: func @broadcast(%[[IN:.*]]: [[IN_T:.*]],  %[[OUT:.*]]: [[OUT_T:.*]]) {
;CHECK:   "xla_lhlo.broadcast_in_dim"(%[[IN]], %[[OUT]])
;CHECK:   {broadcast_dimensions = dense<0> : tensor<1xi64>}
;CHECK:   : ([[IN_T]], [[OUT_T]]) -> ()
;CHECK: }
)");
}

TEST_F(LhloGenTest, Iota) {
  CompileAndVerifyIr(R"(
 HloModule Iota

 ENTRY %Iota() -> s64[10, 5] {
  ROOT %iota = s64[10, 5]{1,0} iota(), iota_dimension=0
})",
                     R"(
;CHECK: func @iota(%[[OUT:.*]]: [[OUT_T:.*]]) {
;CHECK:   "xla_lhlo.iota"(%[[OUT]])
;CHECK:   {iota_dimension = 0 : i64} : ([[OUT_T]]) -> ()
;CHECK: }
)");
}

TEST_F(LhloGenTest, AddReduce) {
  CompileAndVerifyIr(R"(
HloModule AddReduce

%add (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %AddReduce (x: f32[100,10], c: f32[]) -> f32[100] {
  %x = f32[100,10]{1,0} parameter(0)
  %c = f32[] parameter(1)
  ROOT %reduce = f32[100]{0} reduce(f32[100,10]{1,0} %x, f32[] %c), dimensions={1}, to_apply=%add
})",
                     R"(
;CHECK: func @reduce(%[[ARG:.*]]: [[ARGT:.*]], %[[CST:.*]]: memref<f32>, %[[RES:.*]]: [[REST:.*]]) {
;CHECK:   "xla_lhlo.reduce"(%[[ARG]], %[[CST]], %[[RES]]) ( {
;CHECK:   ^bb0(%[[FARG0:.*]]: memref<f32>, %[[FARG1:.*]]: memref<f32>, %[[FRES:.*]]: memref<f32>):
;CHECK:      %[[LHS:.*]] = tensor_load %[[FARG0]] : memref<f32>
;CHECK:      %[[RHS:.*]] = tensor_load %[[FARG1]] : memref<f32>
;CHECK:      %[[RES:.*]] = xla_hlo.add %[[LHS]], %[[RHS]] : tensor<f32>
;CHECK:      tensor_store %[[RES]], %[[FRES]] : memref<f32>
;CHECK:     "xla_lhlo.terminator"() : () -> ()
;CHECK-NEXT: }) {dimensions = dense<1> : tensor<1xi64>} : ([[ARGT]], memref<f32>, [[REST]]) -> ()
      )");
}

TEST_F(LhloGenTest, Abs) {
  CompileAndVerifyIr(R"(
HloModule Abs
ENTRY %Abs (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %abs = f32[2,2]{1,0} abs(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @abs(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.abs"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Ceil) {
  CompileAndVerifyIr(R"(
HloModule Ceil
ENTRY %Ceil (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %ceil = f32[2,2]{1,0} ceil(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @ceil(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.ceil"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Cos) {
  CompileAndVerifyIr(R"(
HloModule Cos
ENTRY %Cos (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %cos = f32[2,2]{1,0} cosine(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @cosine(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.cos"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Neg) {
  CompileAndVerifyIr(R"(
HloModule Neg
ENTRY %Neg (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %neg = f32[2,2]{1,0} negate(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @negate(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.neg"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Rem) {
  CompileAndVerifyIr(R"(
HloModule Rem
ENTRY %Rem(x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %rem = f32[2,2]{1,0} remainder(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @remainder(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.remainder"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Sign) {
  CompileAndVerifyIr(R"(
HloModule Sign
ENTRY %Sign (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %sign = f32[2,2]{1,0} sign(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @sign(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.sign"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

TEST_F(LhloGenTest, Tanh) {
  CompileAndVerifyIr(R"(
HloModule Tanh
ENTRY %Tanh (val: f32[2,2]) -> f32[2,2] {
  %val = f32[2,2]{1,0} parameter(0)
  ROOT %tanh = f32[2,2]{1,0} tanh(f32[2,2]{1,0} %val)
})",
                     R"(
;CHECK: func @tanh(%[[ARG0:.*]]: [[TYPE:.*]], %[[ARG1:.*]]: [[TYPE]]) {
;CHECK:   "xla_lhlo.tanh"(%[[ARG0]], %[[ARG1]]) : ([[TYPE]], [[TYPE]]) -> ()
;CHECK: }
      )");
}

}  // namespace mlir_gpu
}  // namespace xla
