/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/scan_loop_accumulator_input_unification.h"

#include <memory>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/copy_insertion.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ScanLoopAccumulatorInputUnificationTest = HloTestBase;

HloInstruction* GetTopLevelWhileInstruction(HloModule* module) {
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      return instr;
    }
  }
  return nullptr;
}

TEST_F(ScanLoopAccumulatorInputUnificationTest, UnifyAccumulatorInput) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    
    dynamic-slice.0 = s32[1] dynamic-slice(get-tuple-element.54, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.2 = s32[] reshape(dynamic-slice.0)
    add.1 = s32[] add(get-tuple-element.47, reshape.2)

    reshape.3 = s32[1] reshape(add.1)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.3, get-tuple-element.46)
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  outer_body {
    wide.arg_tuple.8 = (s32[], s32[], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2

    constant.3 = s32[] constant(0)
    broadcast = s32[8] broadcast(constant.3), dimensions={}

    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, get-tuple-element.47, broadcast, get-tuple-element.48)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2

    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT out = (s32[], s32[], s32[8]) tuple(add.0, get-tuple-element.47, get-tuple-element.40)
  }
   
  outer_cond {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    tuple.8 = (s32[], s32[], s32[8]) tuple(constant.3, init, array)
    while = (s32[], s32[], s32[8]) while(tuple.8), condition=outer_cond, body=outer_body
    ROOT get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  auto module_clone = module->Clone();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  EXPECT_TRUE(simplified_loop);

  // Index 2 and 3 of the while are replaced with the input arrays.
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      EXPECT_EQ(instr->while_init()->operand(2)->opcode(),
                HloOpcode::kConstant);
    }
  }

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(module), std::move(module_clone), {}, std::nullopt, true));
}

TEST_F(ScanLoopAccumulatorInputUnificationTest, UnifyAccumulatorInput2) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    get-tuple-element.55 = s32[8] get-tuple-element(wide.arg_tuple.8), index=4
    get-tuple-element.56 = s32[8] get-tuple-element(wide.arg_tuple.8), index=5
    
    dynamic-slice.0 = s32[1] dynamic-slice(get-tuple-element.54, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.2 = s32[] reshape(dynamic-slice.0)
    add.1 = s32[] add(get-tuple-element.47, reshape.2)

    reshape.3 = s32[1] reshape(add.1)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.3, get-tuple-element.46)
    
    dynamic-slice.1 = s32[1] dynamic-slice(get-tuple-element.56, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.4 = s32[] reshape(dynamic-slice.1)
    add.2 = s32[] multiply(get-tuple-element.47, reshape.4)

    reshape.5 = s32[1] reshape(add.2)
    dynamic-update-slice.1 = s32[8] dynamic-update-slice(get-tuple-element.55, reshape.5, get-tuple-element.46)
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54, dynamic-update-slice.1, get-tuple-element.56)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  outer_body {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3

    constant.3 = s32[] constant(0)
    broadcast = s32[8] broadcast(constant.3), dimensions={}
    broadcast2 = s32[8] broadcast(constant.3), dimensions={}

    tuple.8 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) tuple(constant.3, get-tuple-element.47, broadcast, get-tuple-element.48, broadcast2, get-tuple-element.54)
    while = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
    get-tuple-element.41 = s32[8] get-tuple-element(while), index=4
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT out = (s32[], s32[], s32[8], s32[8]) tuple(add.0, get-tuple-element.47, get-tuple-element.40, get-tuple-element.41)
  }
   
  outer_cond {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    array2 = s32[8] constant({10,20,30,40,50,60,70,80})
    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, init, array, array2)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=outer_cond, body=outer_body
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
    get-tuple-element.41 = s32[8] get-tuple-element(while), index=3
    ROOT out = (s32[8],s32[8]) tuple(get-tuple-element.40, get-tuple-element.41)
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  auto module_clone = module->Clone();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  EXPECT_TRUE(simplified_loop);

  // Index 2 and 3 of the while are replaced with the input arrays.
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      EXPECT_EQ(instr->while_init()->operand(2)->opcode(),
                HloOpcode::kConstant);
      EXPECT_EQ(instr->while_init()->operand(3)->opcode(),
                HloOpcode::kConstant);
    }
  }

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(module), std::move(module_clone), {}, std::nullopt, true));
}

TEST_F(ScanLoopAccumulatorInputUnificationTest, AccumulatorAllocateOutside) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    
    dynamic-slice.0 = s32[1] dynamic-slice(get-tuple-element.54, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.2 = s32[] reshape(dynamic-slice.0)
    add.1 = s32[] add(get-tuple-element.47, reshape.2)

    reshape.3 = s32[1] reshape(add.1)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.3, get-tuple-element.46)
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  outer_body {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3

    constant.3 = s32[] constant(0)
    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, get-tuple-element.47, get-tuple-element.54, get-tuple-element.48)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2

    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT out = (s32[], s32[], s32[8], s32[8]) tuple(add.0, get-tuple-element.47, get-tuple-element.48, get-tuple-element.40)
  }
   
  outer_cond {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    buffer = s32[8] broadcast(constant.3), dimensions={}
    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, init, array, buffer)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=outer_cond, body=outer_body
    ROOT get-tuple-element.40 = s32[8] get-tuple-element(while), index=3
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  // Buffer is not replaced with input since it is allocated outside the outer
  // loop.
  EXPECT_FALSE(simplified_loop);
}

TEST_F(ScanLoopAccumulatorInputUnificationTest, InputDifferentShape) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8,10]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8,10] get-tuple-element(wide.arg_tuple.8), index=3
  
    zero = s32[] constant(0)
    dynamic-slice.0 = s32[1,10] dynamic-slice(get-tuple-element.54, get-tuple-element.46, zero), dynamic_slice_sizes={1,10}
    reshape.2 = s32[10] reshape(dynamic-slice.0)
  
    dynamic-slice.1 = s32[1] dynamic-slice(reshape.2, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.3 = s32[] reshape(dynamic-slice.1)

    add.1 = s32[] add(get-tuple-element.47, reshape.3)

    reshape.4 = s32[1] reshape(add.1)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.4, get-tuple-element.46)
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8,10]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8,10]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  ENTRY main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8,10] parameter(0)
    broadcast.5 = s32[8] broadcast(constant.3), dimensions={}
    
    tuple.8 = (s32[], s32[], s32[8], s32[8,10]) tuple(constant.3, init, broadcast.5, array)
    while = (s32[], s32[], s32[8], s32[8,10]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.39 = s32[] get-tuple-element(while), index=1
    ROOT get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(ScanLoopAccumulatorInputUnificationTest, MultipleUsersInput) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    // buffer
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    // input with multiple users
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    // buffer
    get-tuple-element.55 = s32[8] get-tuple-element(wide.arg_tuple.8), index=4
    // input
    get-tuple-element.56 = s32[8] get-tuple-element(wide.arg_tuple.8), index=5
    
    // this is here only to have another user for gte.54
    mult = s32[8] multiply(get-tuple-element.54, get-tuple-element.54)
    
    dynamic-slice.0 = s32[1] dynamic-slice(get-tuple-element.54, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.2 = s32[] reshape(dynamic-slice.0)
    add.1 = s32[] add(get-tuple-element.47, reshape.2)

    reshape.3 = s32[1] reshape(add.1)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.3, get-tuple-element.46)
    
    dynamic-slice.1 = s32[1] dynamic-slice(get-tuple-element.56, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.4 = s32[] reshape(dynamic-slice.1)
    add.2 = s32[] multiply(get-tuple-element.47, reshape.4)

    reshape.5 = s32[1] reshape(add.2)
    dynamic-update-slice.1 = s32[8] dynamic-update-slice(get-tuple-element.55, reshape.5, get-tuple-element.46)
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54, dynamic-update-slice.1, get-tuple-element.56)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }
  
  outer_body {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.56 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    
    constant.3 = s32[] constant(0)
    broadcast = s32[8] broadcast(constant.3), dimensions={}
    broadcast2 = s32[8] broadcast(constant.3), dimensions={}

    tuple.8 = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) tuple(constant.3, get-tuple-element.47, broadcast, get-tuple-element.54, broadcast2, get-tuple-element.56)
    while = (s32[], s32[], s32[8], s32[8], s32[8], s32[8]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
    get-tuple-element.41 = s32[8] get-tuple-element(while), index=4

    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT out = (s32[], s32[], s32[8], s32[8]) tuple(add.0, get-tuple-element.47, get-tuple-element.40, get-tuple-element.41)
  }
  
  outer_cond {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  ENTRY main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    array2 = s32[8] constant({10,20,30,40,50,60,70,80})
    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, init, array, array2)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=outer_cond, body=outer_body
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
    get-tuple-element.41 = s32[8] get-tuple-element(while), index=3
    ROOT out = (s32[8],s32[8]) tuple(get-tuple-element.40, get-tuple-element.41)
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  auto module_clone = module->Clone();

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  EXPECT_TRUE(simplified_loop);

  // Only index 2 is replaced with the array.
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      EXPECT_EQ(instr->while_init()->operand(2)->opcode(),
                HloOpcode::kConstant);
    }
  }

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(module), std::move(module_clone), {}, std::nullopt, true));
}

TEST_F(ScanLoopAccumulatorInputUnificationTest,
       UnifyAccumulatorInputCheckCopy) {
  [[maybe_unused]] constexpr char kModule[] = R"(
  HloModule jit_scan

  wide.region_0.7 {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[8], s32[10]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.54 = s32[8] get-tuple-element(wide.arg_tuple.8), index=3
    get-tuple-element.55 = s32[10] get-tuple-element(wide.arg_tuple.8), index=4
    dynamic-slice.0 = s32[1] dynamic-slice(get-tuple-element.54, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.2 = s32[] reshape(dynamic-slice.0)
    dynamic-slice.1 = s32[1] dynamic-slice(get-tuple-element.55, get-tuple-element.46), dynamic_slice_sizes={1}
    reshape.3 = s32[] reshape(dynamic-slice.1)
    add.1 = s32[] add(reshape.3, reshape.2)
    add.2 = s32[] add(add.1, get-tuple-element.47)
    
    reshape.4 = s32[1] reshape(add.2)
    dynamic-update-slice.0 = s32[8] dynamic-update-slice(get-tuple-element.48, reshape.4, get-tuple-element.46)
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT tuple.10 = (s32[], s32[], s32[8], s32[8], s32[10]) tuple(add.0, add.1, dynamic-update-slice.0, get-tuple-element.54, get-tuple-element.55)
  } // wide.region_0.7

  wide.region_1.29 {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[8], s32[10]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }
  
  outer_body {
    wide.arg_tuple.8 = (s32[], s32[], s32[8], s32[10]) parameter(0)
    get-tuple-element.46 = s32[] get-tuple-element(wide.arg_tuple.8), index=0
    get-tuple-element.47 = s32[] get-tuple-element(wide.arg_tuple.8), index=1
    get-tuple-element.48 = s32[8] get-tuple-element(wide.arg_tuple.8), index=2
    get-tuple-element.55 = s32[10] get-tuple-element(wide.arg_tuple.8), index=3
    
    constant.3 = s32[] constant(0)
    broadcast = s32[8] broadcast(constant.3), dimensions={}
    
    tuple.8 = (s32[], s32[], s32[8], s32[8], s32[10]) tuple(constant.3, get-tuple-element.47, broadcast, get-tuple-element.48, get-tuple-element.55)
    while = (s32[], s32[], s32[8], s32[8], s32[10]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
    
    const = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.46, const)
    ROOT out = (s32[], s32[], s32[8], s32[10]) tuple(add.0, get-tuple-element.47, get-tuple-element.40, get-tuple-element.55)
  }
  
  outer_cond {
    constant.5 = s32[] constant(8)
    wide.arg_tuple.30 = (s32[], s32[], s32[8], s32[10]) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(wide.arg_tuple.30), index=0
    ROOT compare.0 = pred[] compare(get-tuple-element.16, constant.5), direction=LT
  }

  ENTRY main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    other_input = s32[10] constant({10,20,30,40,50,60,70,80,90,100})
    tuple.8 = (s32[], s32[], s32[8], s32[10]) tuple(constant.3, init, array, other_input)
    while = (s32[], s32[], s32[8], s32[10]) while(tuple.8), condition=outer_cond, body=outer_body
    get-tuple-element.39 = s32[8] get-tuple-element(while), index=2
    get-tuple-element.40 = s32[10] get-tuple-element(while), index=3
    ROOT out = (s32[8],s32[10]) tuple(get-tuple-element.39, get-tuple-element.40)
  } // main.43
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();

  // Check the inserted copies before applying the copy insertion pass.
  auto module_clone = module->Clone();
  TF_ASSERT_OK_AND_ASSIGN(bool clone_copy_inserted,
                          CopyInsertion().Run(module_clone.get()));
  EXPECT_TRUE(clone_copy_inserted);
  HloInstruction* while_instruction =
      GetTopLevelWhileInstruction(module_clone.get());
  EXPECT_EQ(
      while_instruction->while_body()->root_instruction()->operand(2)->opcode(),
      HloOpcode::kCopy);

  TF_ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      ScanLoopAccumulatorInputUnification().Run(module.get()));
  EXPECT_TRUE(simplified_loop);

  // Check the inserted copies after applying the copy insertion pass and
  // removing double buffers.
  TF_ASSERT_OK_AND_ASSIGN(bool copy_inserted,
                          CopyInsertion().Run(module.get()));
  EXPECT_TRUE(copy_inserted);
  VLOG(3) << "After copy_insertion:\n" << module->ToString();
  while_instruction = GetTopLevelWhileInstruction(module.get());
  EXPECT_NE(
      while_instruction->while_body()->root_instruction()->operand(2)->opcode(),
      HloOpcode::kCopy);
}

}  // namespace
}  // namespace xla
