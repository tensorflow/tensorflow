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

#include "tensorflow/compiler/xla/service/scatter_expander.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class ScatterExpanderTest : public HloTestBase {};

TEST_F(ScatterExpanderTest, ScatterOperandWithoutLayout) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      parameter0 = s32[] parameter(0)
      ROOT parameter1 = s32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = s32[5] iota(), iota_dimension=0
      indices = s32[1] parameter(0)
      update = s32[] constant(0)
      ROOT scatter = s32[5]{0} scatter(operand, indices, update),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=0,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // The HLO parser changes all no layout shapes from the input to have a
  // default layout. Clear the layout of the scatter operand for testing.
  HloInstruction* scatter_operand = FindInstruction(module.get(), "operand");
  scatter_operand->mutable_shape()->clear_layout();

  ScatterExpander scatter_expander(ScatterExpander::kEliminateAllScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterExpanderTest, EliminateSimpleScattersSkipsNontrivialScatter) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      parameter0 = s32[] parameter(0)
      ROOT parameter1 = s32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=scatter_computation,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // The HLO parser changes all no layout shapes from the input to have a
  // default layout. Clear the layout of the scatter operand for testing.
  HloInstruction* scatter_operand = FindInstruction(module.get(), "operand");
  scatter_operand->mutable_shape()->clear_layout();

  ScatterExpander scatter_expander(ScatterExpander::kEliminateSimpleScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(ScatterExpanderTest, EliminateSimpleScattersRewritesTrivialScatter) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      parameter0 = s32[] parameter(0)
      ROOT parameter1 = s32[] parameter(1)
    }

    ENTRY kernel_entry {
      operand = s32[5] iota(), iota_dimension=0
      indices = s32[1] parameter(0)
      update = s32[] constant(0)
      ROOT scatter = s32[5]{0} scatter(operand, indices, update),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=0,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // The HLO parser changes all no layout shapes from the input to have a
  // default layout. Clear the layout of the scatter operand for testing.
  HloInstruction* scatter_operand = FindInstruction(module.get(), "operand");
  scatter_operand->mutable_shape()->clear_layout();

  ScatterExpander scatter_expander(ScatterExpander::kEliminateSimpleScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

}  // namespace
}  // namespace xla
