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

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class ScatterExpanderTest : public HloTestBase {
 protected:
  // The HLO parser changes all no layout shapes from the input to have a
  // default layout. Clear the layout of the scatter operand for testing.
  void ClearInstructionLayout(HloModule* module, absl::string_view inst_name) {
    HloInstruction* inst = FindInstruction(module, inst_name);
    inst->mutable_shape()->clear_layout();
  }
};

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

  ClearInstructionLayout(module.get(), "operand");

  ScatterExpander scatter_expander(ScatterExpander::kEliminateAllScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterExpanderTest, ScatterMultipleOperandsWithoutLayout) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      p0 = s32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT tuple = tuple(p2, p3)
    }

    ENTRY kernel_entry {
      operand0 = s32[5] iota(), iota_dimension=0
      operand1 = f32[5] constant({2,4,6,8,10})
      indices = s32[1] parameter(0)
      update0 = s32[] constant(0)
      update1 = f32[] constant(1)
      ROOT scatter = (s32[5]{0}, f32[5]{0}) scatter(operand0, operand1, indices, update0, update1),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=0,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ClearInstructionLayout(module.get(), "operand0");
  ClearInstructionLayout(module.get(), "operand1");

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

  ClearInstructionLayout(module.get(), "operand");

  ScatterExpander scatter_expander(ScatterExpander::kEliminateSimpleScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(ScatterExpanderTest,
       EliminateSimpleMultioutpuScattersSkipsNontrivialScatter) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      p0 = s32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT tuple = tuple(p2, p3)
    }

    ENTRY kernel_entry {
      operand0 = s32[3,3] parameter(0)
      operand1 = bf16[3,3] parameter(1)
      indices = s32[2] parameter(2)
      update0 = s32[2,3] parameter(3)
      update1 = bf16[2,3] parameter(4)
      ROOT scatter = (s32[3,3], bf16[3,3]) scatter(operand0, operand1, indices, update0, update1),
          to_apply=scatter_computation,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ClearInstructionLayout(module.get(), "operand0");
  ClearInstructionLayout(module.get(), "operand1");

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

  ClearInstructionLayout(module.get(), "operand");

  ScatterExpander scatter_expander(ScatterExpander::kEliminateSimpleScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterExpanderTest,
       EliminateSimpleMultioutputScattersRewritesTrivialScatter) {
  const char* kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      p0 = s32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = f32[] parameter(3)
      ROOT tuple = tuple(p2, p3)
    }

    ENTRY kernel_entry {
      operand0 = s32[5] iota(), iota_dimension=0
      operand1 = f32[5] iota(), iota_dimension=0
      indices = s32[1] parameter(0)
      update0 = s32[] constant(0)
      update1 = f32[] constant(0)
      ROOT scatter = (s32[5]{0}, f32[5]{0}) scatter(operand0, operand1, indices, update0, update1),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=0,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ClearInstructionLayout(module.get(), "operand0");
  ClearInstructionLayout(module.get(), "operand1");

  ScatterExpander scatter_expander(ScatterExpander::kEliminateSimpleScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterExpanderTest, DoNotEliminateScatterWithAssociativeCombiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      arg1.173 = s32[] parameter(1)
      arg0.172 = s32[] parameter(0)
      ROOT add.48 = s32[] add(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = s32[1,4096] parameter(0)
      pad.96 = s32[4096,2] parameter(1)
     bitcast.2748 = s32[4096,1,1] parameter(2)
      ROOT scatter.48 = s32[1,4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterExpander scatter_expander(
      ScatterExpander::kEliminateIndeterminisitcScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(ScatterExpanderTest, EliminateScatterWithNonAssociativeCombiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT add.48 = f32[] add(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = f32[1,4096] parameter(0)
      pad.96 = s32[4096,2] parameter(1)
     bitcast.2748 = f32[4096,1,1] parameter(2)
      ROOT scatter.48 = f32[1,4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterExpander scatter_expander(
      ScatterExpander::kEliminateIndeterminisitcScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterExpanderTest, DoNotEliminateScatterWithAssociativeFp32Combiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT max.48 = f32[] maximum(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = f32[1,4096] parameter(0)
      pad.96 = s32[4096,2] parameter(1)
     bitcast.2748 = f32[4096,1,1] parameter(2)
      ROOT scatter.48 = f32[1,4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterExpander scatter_expander(
      ScatterExpander::kEliminateIndeterminisitcScatters);
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&scatter_expander, module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace xla
