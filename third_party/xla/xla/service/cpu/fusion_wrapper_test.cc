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
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);

  // A subsequent run should be a no-op -- the scatter is already fused.
  TF_ASSERT_OK_AND_ASSIGN(changed, wrapper.Run(m.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
