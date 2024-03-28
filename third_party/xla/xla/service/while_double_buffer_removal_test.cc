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

#include "xla/service/while_double_buffer_removal.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"

namespace xla {
namespace {

using WhileDoubleBufferRemovalTest = HloTestBase;

TEST_F(WhileDoubleBufferRemovalTest, RemoveDoubleBuffer) {
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

  ENTRY main.43 {
    constant.3 = s32[] constant(0)
    init = s32[] constant(0)
    array = s32[8] constant({1,2,3,4,5,6,7,8})
    broadcast.5 = s32[8] broadcast(constant.3), dimensions={}
    tuple.8 = (s32[], s32[], s32[8], s32[8]) tuple(constant.3, init, broadcast.5, array)
    while = (s32[], s32[], s32[8], s32[8]) while(tuple.8), condition=wide.region_1.29, body=wide.region_0.7
    get-tuple-element.39 = s32[] get-tuple-element(while), index=1
    ROOT get-tuple-element.40 = s32[8] get-tuple-element(while), index=2
  } // main.43
  
  )";

  auto module = ParseAndReturnVerifiedModule(kModule).value();

  for (HloComputation* comp : module->computations()) {
    if (comp->IsWhileBodyComputation()) {
      EXPECT_EQ(comp->root_instruction()->operand_count(), 4);
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileDoubleBufferRemoval().Run(module.get()));
  EXPECT_TRUE(simplified_loop);
  for (HloComputation* comp : module->computations()) {
    if (comp->IsWhileBodyComputation()) {
      EXPECT_EQ(comp->root_instruction()->operand_count(), 3);
    }
  }
}

}  // namespace
}  // namespace xla
