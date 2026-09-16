/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/scatter_expander.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {

using ScatterExpanderTest = HloHardwareIndependentTestBase;

TEST_F(ScatterExpanderTest, SupportedScatter) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    update_computation {
      %operand = f32[] parameter(0)
      %update = f32[] parameter(1)
      ROOT %add = f32[] add(%operand, %update)
    }
    ENTRY entry {
      %operand = f32[32,32] parameter(0)
      %indices = s32[2] parameter(1)
      %update = f32[2,2] parameter(2)
      ROOT %scatter = f32[32,32] scatter(f32[32,32] %operand, s32[2] %indices, f32[2,2] %update),
        update_window_dims={0,1},
        scatter_dims_to_operand_dims={0,1},
        inserted_window_dims={},
        index_vector_dim=0,
        to_apply=update_computation
    })"));

  GpuScatterExpander scatter_expander;
  ASSERT_OK_AND_ASSIGN(bool changed, scatter_expander.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScatterExpanderTest, UnsupportedElementType) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    update_computation {
      %operand = c128[] parameter(0)
      %update = c128[] parameter(1)
      ROOT %add = c128[] add(%operand, %update)
    }
    ENTRY entry {
      %operand = c128[32,32] parameter(0)
      %indices = s32[2] parameter(1)
      %update = c128[2,2] parameter(2)
      ROOT %scatter = c128[32,32] scatter(c128[32,32] %operand, s32[2] %indices, c128[2,2] %update),
        update_window_dims={0,1},
        scatter_dims_to_operand_dims={0,1},
        inserted_window_dims={},
        index_vector_dim=0,
        to_apply=update_computation
    })"));

  GpuScatterExpander scatter_expander;
  ASSERT_OK_AND_ASSIGN(bool changed, scatter_expander.Run(module.get()));
  // Scatter with unsupported element type is expanded.
  // c128 has 128 bits per element, maximum supported is 64 bits.
  EXPECT_TRUE(changed);
}

TEST_F(ScatterExpanderTest, SupportedVariadicScatter) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    override {
      %p0 = f32[] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = f32[] parameter(2)
      %p3 = s32[] parameter(3)
      ROOT %override = tuple(%p2, %p3)
    }

    ENTRY entry {
      %operand_1 = f32[50,5]  parameter(0)
      %operand_2 = s32[50,5]  parameter(1)
      %indices =  s32[24,1] iota(), iota_dimension=0
      %update_1 = f32[24,1,3] parameter(2)
      %update_2 = s32[24,1,3] parameter(3)
      ROOT %scatter = (f32[50,5], s32[50,5]) scatter(
          %operand_1, %operand_2, %indices, %update_1, %update_2),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=true,
        to_apply=override
    })"));

  GpuScatterExpander scatter_expander;
  ASSERT_OK_AND_ASSIGN(bool changed, scatter_expander.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScatterExpanderTest, UnsupportedVariadicScatter) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    override {
      %p0 = f32[] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = f32[] parameter(2)
      %p3 = s32[] parameter(3)
      ROOT %override = tuple(%p2, %p3)
    }

    ENTRY entry {
      %operand_1 = f32[50,5]  parameter(0)
      %operand_2 = s32[50,5]  parameter(1)
      %indices =  s32[24,1] iota(), iota_dimension=0
      %update_1 = f32[24,1,3] parameter(2)
      %update_2 = s32[24,1,3] parameter(3)
      ROOT %scatter = (f32[50,5], s32[50,5]) scatter(
          %operand_1, %operand_2, %indices, %update_1, %update_2),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=false,
        to_apply=override
    })"));

  GpuScatterExpander scatter_expander;
  ASSERT_OK_AND_ASSIGN(bool changed, scatter_expander.Run(module.get()));
  // Variadic scatter with non-unique indices is not expanded.
  EXPECT_TRUE(changed);
}

}  // namespace xla
