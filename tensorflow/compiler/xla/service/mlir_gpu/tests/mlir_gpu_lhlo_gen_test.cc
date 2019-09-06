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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_irgen_test_base.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace mlir_gpu {

class LhloGenTest : public MlirIrGenTestBase {};

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
;CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ARG0]][[INDEX:.*]]
;CHECK: %[[VAL0:.*]] = llvm.load %[[GEP0]]
;CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][[INDEX]]
;CHECK: %[[VAL1:.*]] = llvm.load %[[GEP1]]
;CHECK: %[[VAL2:.*]] = llvm.fadd %[[VAL0]], %[[VAL1]]
;CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[ARG2]][[INDEX]]
;CHECK: llvm.store %[[VAL2]], %[[GEP2]]
      )",
                     LoweringStage::LLVM);
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
;CHECK:   %[[REF1:.*]] = tensor_load %[[ARG1]] : [[TYPE:.*]]
;CHECK:   %[[REF2:.*]] = tensor_load %[[ARG2]] : [[TYPE]]
;CHECK:   %[[ADD:.*]] = "xla_hlo.add"(%[[REF1]], %[[REF2]]) {name = "add"}
;CHECK:   %[[REF0:.*]] = tensor_load %[[ARG0]] : [[TYPE]]
;CHECK:   %[[MUL:.*]] = "xla_hlo.mul"(%[[ADD]], %[[REF0]]) {name = "multiply"}
;CHECK:   tensor_store %[[MUL]], %[[RESULT]]
;CHECK:   "xla_lhlo.terminator"()
;CHECK-NEXT: }
      )");
}

}  // namespace mlir_gpu
}  // namespace xla
