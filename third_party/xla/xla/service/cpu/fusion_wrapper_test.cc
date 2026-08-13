/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/fusion_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

class FusionWrapperTest : public HloHardwareIndependentTestBase {};

TEST_F(FusionWrapperTest, Scatter) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT sum = f32[] add(p0, p1)
    }
    ENTRY e {
      operand = f32[10,5] parameter(0)
      indices = s32[24,1] parameter(1)
      update = f32[24,2,3] parameter(2)
      ROOT scatter = f32[10,5] scatter(
          f32[10,5] operand,
          s32[24,1] indices,
          f32[24,2,3] update
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=false,
        to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(false, false);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);

  // A subsequent run should be a no-op -- the scatter is already fused.
  ASSERT_OK_AND_ASSIGN(changed, wrapper.Run(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(FusionWrapperTest, TransposeWrappedWithNewFusionEmitters) {
  // Standalone transposes route to ElementalKernelEmitter when unwrapped.
  // Wrap them when the new fusion emitters are enabled.
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32] parameter(0)
      ROOT t = f32[32,64] transpose(p0), dimensions={1,0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(/*using_new_fusion_emitter=*/true,
                        /*use_tiled_emitter=*/false);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(FusionWrapperTest, DynamicUpdateSliceWrappedWithNewFusionEmitters) {
  // The MLIR fusion path has a dedicated dynamic-update-slice emitter with an
  // in-place check, so wrap standalone dynamic-update-slice when the new fusion
  // emitters are enabled.
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,64] parameter(0)
      p1 = f32[8,8] parameter(1)
      i = s32[] parameter(2)
      j = s32[] parameter(3)
      ROOT dus = f32[64,64] dynamic-update-slice(p0, p1, i, j)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(/*using_new_fusion_emitter=*/true,
                        /*use_tiled_emitter=*/false);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(FusionWrapperTest, MissingElementalOpcodesWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view kUnaryOpcodes[] = {
      "acos", "acosh", "asin", "asinh", "atanh", "cosh", "sinh"};
  for (absl::string_view op : kUnaryOpcodes) {
    std::string hlo_string = absl::StrFormat(R"(
    HloModule m
      ENTRY e {
        p0 = f32[64,32] parameter(0)
        ROOT r = f32[64,32] %s(p0)
      }
    )",
                                             op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper(/*using_new_fusion_emitter=*/true,
                          /*use_tiled_emitter=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_TRUE(changed) << "Failed for opcode: " << op;
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kFusion)
        << "Failed for opcode: " << op;
  }

  {
    static constexpr absl::string_view hlo_string = R"(
    HloModule m
      ENTRY e {
        p0 = s32[64,32] parameter(0)
        p1 = s32[64,32] parameter(1)
        ROOT r = s32[64,32] mulhi(p0, p1)
      }
    )";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper(/*using_new_fusion_emitter=*/true,
                          /*use_tiled_emitter=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_TRUE(changed) << "Failed for opcode: mulhi";
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kFusion);
  }
}

TEST_F(FusionWrapperTest,
       MissingElementalOpcodesNotWrappedWithoutNewFusionEmitters) {
  static constexpr absl::string_view kUnaryOpcodes[] = {
      "acos", "acosh", "asin", "asinh", "atanh", "cosh", "sinh"};
  for (absl::string_view op : kUnaryOpcodes) {
    std::string hlo_string = absl::StrFormat(R"(
    HloModule m
      ENTRY e {
        p0 = f32[64,32] parameter(0)
        ROOT r = f32[64,32] %s(p0)
      }
    )",
                                             op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper(/*using_new_fusion_emitter=*/false,
                          /*use_tiled_emitter=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_FALSE(changed) << "Failed for opcode: " << op;
  }

  {
    static constexpr absl::string_view hlo_string = R"(
    HloModule m
      ENTRY e {
        p0 = s32[64,32] parameter(0)
        p1 = s32[64,32] parameter(1)
        ROOT r = s32[64,32] mulhi(p0, p1)
      }
    )";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper(/*using_new_fusion_emitter=*/false,
                          /*use_tiled_emitter=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_FALSE(changed) << "Failed for opcode: mulhi";
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
