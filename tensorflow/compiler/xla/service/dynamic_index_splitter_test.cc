/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
class DynamicIndexSplitterTest : public HloTestBase {};

TEST_F(DynamicIndexSplitterTest, DynamicSlice) {
  const char* const kDynamicSlice = R"(
    HloModule DynamicSlice_module

    ENTRY entry (operand: s32[4,5,6], indices: s32[3]) -> s32[1,1,1] {
      operand = s32[4,5,6] parameter(0)
      indices = s32[3] parameter(1)
      ROOT dynamic-slice = s32[1,1,1] dynamic-slice(operand, indices), dynamic_slice_sizes={1,1,1}
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_allow_scalar_index_dynamic_ops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kDynamicSlice, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          DynamicIndexSplitter().Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::Parameter(0),
                               op::Reshape(op::Slice(op::Parameter(1))),
                               op::Reshape(op::Slice(op::Parameter(1))),
                               op::Reshape(op::Slice(op::Parameter(1)))));

  for (int i = 0; i < 3; ++i) {
    const HloInstruction* slice = module->entry_computation()
                                      ->root_instruction()
                                      ->operand(i + 1)
                                      ->operand(0);
    EXPECT_EQ(slice->slice_starts(0), i);
    EXPECT_EQ(slice->slice_limits(0), i + 1);
  }
}

TEST_F(DynamicIndexSplitterTest, DynamicUpdateSlice) {
  const char* const kDynamicUpdateSlice = R"(
    HloModule DynamicUpdatedSlice_module

    ENTRY entry (operand: s32[4,5,6], indices: s32[3], update: s32[1,1,1]) -> s32[4,5,6] {
      operand = s32[4,5,6] parameter(0)
      indices = s32[3] parameter(1)
      update = s32[1,1,1] parameter(2)
      ROOT dynamic-update-slice = s32[4,5,6] dynamic-update-slice(operand, update, indices)
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_allow_scalar_index_dynamic_ops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseHloString(kDynamicUpdateSlice, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          DynamicIndexSplitter().Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicUpdateSlice(op::Parameter(0), op::Parameter(2),
                                     op::Reshape(op::Slice(op::Parameter(1))),
                                     op::Reshape(op::Slice(op::Parameter(1))),
                                     op::Reshape(op::Slice(op::Parameter(1)))));

  for (int i = 0; i < 3; ++i) {
    const HloInstruction* slice = module->entry_computation()
                                      ->root_instruction()
                                      ->operand(i + 2)
                                      ->operand(0);
    EXPECT_EQ(slice->slice_starts(0), i);
    EXPECT_EQ(slice->slice_limits(0), i + 1);
  }
}

TEST_F(DynamicIndexSplitterTest, AlreadyScalar) {
  const char* const kDynamicSlice = R"(
    HloModule DynamicSlice_module

    ENTRY entry (operand: s32[4,5,6], index.0: s32[], index.1: s32[], index.2: s32[]) -> s32[1,1,1] {
      operand = s32[4,5,6] parameter(0)
      index.0 = s32[] parameter(1)
      index.1 = s32[] parameter(2)
      index.2 = s32[] parameter(3)
      ROOT dynamic-slice = s32[1,1,1] dynamic-slice(operand, index.0, index.1, index.2), dynamic_slice_sizes={1,1,1}
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_allow_scalar_index_dynamic_ops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kDynamicSlice, config));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          DynamicIndexSplitter().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::DynamicSlice(op::Parameter(0), op::Parameter(1),
                               op::Parameter(2), op::Parameter(3)));
}

}  // namespace
}  // namespace xla
