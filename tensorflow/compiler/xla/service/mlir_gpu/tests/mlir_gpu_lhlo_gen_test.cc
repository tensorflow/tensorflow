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
;CHECK:   "xla_lhlo.add"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) {name = "add"} : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
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
;CHECK:   "xla_lhlo.exp"(%[[ARG0]], %[[ARG1]]) {name = "exponential"} : ([[TYPE]], [[TYPE]]) -> ()
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
;CHECK: load %[[ARG0]][[INDEX:.*]]
;CHECK: load %[[ARG1]][[INDEX]]
;CHECK: store %{{.*}}, %[[ARG2]][[INDEX]]
      )",
                     LoweringStage::GPU);
}

TEST_F(LhloGenTest, AddInLVVMDialect) {
  CompileAndVerifyIr(R"(
HloModule Add

ENTRY %Add (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
  %x = f32[2,2]{1,0} parameter(0)
  %y = f32[2,2]{1,0} parameter(1)
  ROOT %add = f32[2,2]{1,0} add(f32[2,2]{1,0} %x, f32[2,2]{1,0} %y)
})",
                     R"(
;CHECK: func @add_kernel(%[[ARG0:.*]]: [[TYPE:!llvm<.*]], %[[ARG1:.*]]: [[TYPE]], %[[ARG2:.*]]: [[TYPE]]
;CHECK: %[[LD0:.*]] = llvm.load %[[ARG0]] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[LD1:.*]] = llvm.load %[[ARG1]] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[LD2:.*]] = llvm.load %[[ARG2]] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[PTR0:.*]] = llvm.extractvalue %[[LD0]][0 : index]
;CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[PTR0]]
;CHECK: %[[VAL0:.*]] = llvm.load %[[GEP0]]
;CHECK: %[[PTR1:.*]] = llvm.extractvalue %[[LD1]][0 : index]
;CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[PTR1]]
;CHECK: %[[VAL1:.*]] = llvm.load %[[GEP1]]
;CHECK: %[[VAL2:.*]] = llvm.fadd %[[VAL0]], %[[VAL1]]
;CHECK: %[[PTR2:.*]] = llvm.extractvalue %[[LD2]][0 : index]
;CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[PTR2]]
;CHECK: llvm.store %[[VAL2]], %[[GEP2]]
      )",
                     LoweringStage::LLVM);
}

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
;CHECK: %[[CST0:.*]] = llvm.mlir.constant(0 : i64)
;CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ARG0]][%[[CST0]]]
;CHECK: %[[BC0:.*]] = llvm.bitcast %[[GEP0]] : !llvm<"i8*"> to !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[CST1:.*]] = llvm.mlir.constant(0 : i64)
;CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][%[[CST1]]]
;CHECK: %[[BC1:.*]] = llvm.bitcast %[[GEP1]] : !llvm<"i8*"> to !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[CST2:.*]] = llvm.mlir.constant(0 : i64)
;CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[ARG2]][%[[CST2]]]
;CHECK: %[[BC2:.*]] = llvm.bitcast %[[GEP2]] : !llvm<"i8*"> to !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
;CHECK: %[[VL0:.*]] = llvm.load %[[BC0]]
;CHECK: %[[VL1:.*]] = llvm.load %[[BC1]]
;CHECK: %[[VL2:.*]] = llvm.load %[[BC2]]
;CHECK: %[[EV0:.*]] = llvm.extractvalue %[[VL0]][0 : index]
;CHECK: %[[VGEP0:.*]] = llvm.getelementptr %[[EV0]]
;CHECK: %[[VAL0:.*]] = llvm.load %[[VGEP0]]
;CHECK: %[[EV1:.*]] = llvm.extractvalue %[[VL1]][0 : index]
;CHECK: %[[VGEP1:.*]] = llvm.getelementptr %[[EV1]]
;CHECK: %[[VAL1:.*]] = llvm.load %[[VGEP1]]
;CHECK: %[[VAL2:.*]] = llvm.fadd %[[VAL0]], %[[VAL1]]
;CHECK: %[[EV2:.*]] = llvm.extractvalue %[[VL2]][0 : index]
;CHECK: %[[SGEP:.*]] = llvm.getelementptr %[[EV2]]
;CHECK: llvm.store %[[VAL2]], %[[SGEP]]
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
;CHECK:   %[[REF1:.*]] = tensor_load %[[ARG1]] : [[TYPE:.*]]
;CHECK:   %[[REF2:.*]] = tensor_load %[[ARG2]] : [[TYPE]]
;CHECK:   %[[ADD:.*]] = "xla_hlo.add"(%[[REF1]], %[[REF2]]) {name = "add"}
;CHECK:   %[[MUL:.*]] = "xla_hlo.mul"(%[[ADD]], %[[REF0]]) {name = "multiply"}
;CHECK:   tensor_store %[[MUL]], %[[RESULT]]
;CHECK:   "xla_lhlo.terminator"()
;CHECK-NEXT: }
      )");
}

TEST_F(LhloGenTest, FusedReduce) {
  CompileAndVerifyIr(R"(
HloModule FusedReduce

%add (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%fused_computation (param: f32[100,10]) -> f32[10] {
  %param = f32[100,10] parameter(0)
  %constant = f32[] constant(0)
  ROOT %reduce = f32[10]{0} reduce(f32[100,10]{1,0} %param, f32[] %constant), dimensions={0}, to_apply=%add
}

ENTRY %FusedReduce (x: f32[100,10]) -> f32[10] {
  %x = f32[100,10] parameter(0)
  ROOT %fusion = f32[10]{0} fusion(f32[100,10]{1,0} %x), kind=kInput, calls=%fused_computation
}
)",
                     R"(
;CHECK: func @fusion(%[[ARG0:.*]]: [[TYPE:.*]], %[[RESULT:.*]]: [[RTYPE:.*]])
;CHECK: "xla_lhlo.fusion"() ( {
;CHECK:   %[[REF0:.*]] = tensor_load %arg0 : [[TYPE]]
;CHECK:   %[[CT0:.*]] = "xla_hlo.constant"()
;CHECK:   %[[RED:.*]] = "xla_hlo.reduce"(%0, %1) ( {
;CHECK:     ^bb0(%[[BARG0:.*]]: [[ETYPE:.*]], %[[BARG1:.*]]: [[ETYPE]])
;CHECK:       %[[ADD:.*]] = "xla_hlo.add"(%[[BARG0]], %[[BARG1]]) {name = "add"} : ([[ETYPE]], [[ETYPE]]) -> [[ETYPE]]
;CHECK:       "xla_hlo.return"(%[[ADD]])
;CHECK:     })
;CHECK:   tensor_store %[[RED]], %[[RESULT]] : [[RTYPE]]
;CHECK:   "xla_lhlo.terminator"()
;CHECK-NEXT: })
      )");
}

}  // namespace mlir_gpu
}  // namespace xla
