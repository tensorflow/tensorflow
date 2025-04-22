/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/scatter_slice_simplifier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

using ScatterSliceSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(ScatterSliceSimplifierTest, Scatter1D) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %slice = f32[8] slice(%scatter), slice={[0:8]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Scatter(m::Slice(m::Constant()), m::Parameter(0),
                                    m::Parameter(1))
                             .WithShape(F32, {8})));
}

TEST_F(ScatterSliceSimplifierTest, Scatter3D) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY main {
  %indices = s32[2] parameter(0)
  %updates = f32[2,4,4] parameter(1)
  %operands = f32[5,4,4] constant(0)
  %scatter = f32[5,4,4] scatter(%operands, %indices, %updates), update_window_dims={1,2}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %slice = f32[4,4,4] slice(%scatter), slice={[0:4], [0:4], [0:4]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Scatter(m::Slice(m::Constant()), m::Parameter(0),
                                    m::Parameter(1))
                             .WithShape(F32, {4, 4, 4})));
}

TEST_F(ScatterSliceSimplifierTest, ScatterMultiOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32_add_F16 {
  %lhs.0 = f32[] parameter(0)
  %rhs.0 = f32[] parameter(2)
  %add.0 = f32[] add(%lhs.0, %rhs.0)
  %lhs.1 = f16[] parameter(1)
  %rhs.1 = f16[] parameter(3)
  %add.1 = f16[] add(%lhs.1, %rhs.1)
  ROOT %tuple = (f32[], f16[]) tuple(%add.0, %add.1)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates.0 = f32[4] parameter(1)
  %updates.1 = f16[4] parameter(2)
  %operands.0 = f32[9] constant(0)
  %operands.1 = f16[9] constant(0)
  %scatter = (f32[9], f16[9]) scatter(%operands.0, %operands.1, %indices, %updates.0, %updates.1), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32_add_F16
  %gte.0 = f32[9] get-tuple-element(%scatter), index=0
  %slice.0 = f32[8] slice(%gte.0), slice={[0:8]}
  %gte.1 = f16[9] get-tuple-element(%scatter), index=1
  %slice.1 = f16[8] slice(%gte.1), slice={[0:8]}
  ROOT %tuple = (f32[8], f16[8]) tuple(%slice.0, %slice.1)
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  auto expected_scatter =
      m::Scatter(m::Slice(m::Constant()), m::Slice(m::Constant()),
                 m::Parameter(0), m::Parameter(1), m::Parameter(2));

  Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(F16, {8})});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(expected_scatter),
                                  m::GetTupleElement(expected_scatter))
                             .WithShapeEqualTo(&expected_shape)));
}

TEST_F(ScatterSliceSimplifierTest, NotMatching) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

slice_not_truncation {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %slice = f32[8] slice(%scatter), slice={[1:9]}
}

slice_with_stride {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %slice = f32[4] slice(%scatter), slice={[0:8:2]}
}

scatter_multiple_users {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  %slice = f32[8] slice(%scatter), slice={[0:8]}
  ROOT %tuple = (f32[9], f32[8]) tuple(%scatter, %slice)
}

scatter_incompatible_slices {
  %indices = s32[2] parameter(0)
  %updates = f32[2,4] parameter(1)
  %operands = f32[4,4] constant(0)
  %scatter = f32[4,4] scatter(%operands, %indices, %updates), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  %slice.0 = f32[3,4] slice(%scatter), slice={[0:3], [0:4]}
  %slice.1 = f32[4,3] slice(%scatter), slice={[0:4], [0:3]}
  ROOT %tuple = (f32[3,4], f32[4,3]) tuple(%slice.0, %slice.1)
}

slice_not_found {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[8] constant(0)
  %scatter = f32[8] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %exp = f32[8] exponential(%scatter)
}

slice_update_dimensions {
  %indices = s32[10] parameter(0)
  %updates = f32[10,1,128] parameter(1)
  %operands = f32[100,128] constant(0)
  %scatter = f32[100,128] scatter(%operands, %indices, %updates), update_window_dims={1,2}, inserted_window_dims={}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  ROOT %slice = f32[100,64] slice(%scatter), slice={[0:100], [0:64]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_FALSE(RunHloPass(&test_pass, module.get()).value());
}

TEST_F(ScatterSliceSimplifierTest, IntermediaryUsers) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  %unary = f32[9] abs(%scatter)
  %slice.0 = f32[8] slice(%unary), slice={[0:8]}
  %binary = f32[9] maximum(%scatter, %operands)
  %slice.1 = f32[8] slice(%binary), slice={[0:8]}
  ROOT %tuple = (f32[8], f32[8]) tuple(%slice.0, %slice.1)
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  auto expected_scatter =
      m::Scatter(m::Slice(m::Constant()), m::Parameter(0), m::Parameter(1));

  Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(F32, {8})});
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Abs(expected_scatter),
                          m::Maximum(expected_scatter, m::Slice(m::Constant())))
                     .WithShapeEqualTo(&expected_shape)));
}

TEST_F(ScatterSliceSimplifierTest, IntermediaryChain) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  %elementwise.0 = f32[9] abs(%scatter)
  %elementwise.1 = f32[9] exponential(%elementwise.0)
  %elementwise.2 = f32[9] add(%elementwise.0, %elementwise.1)
  ROOT %result = f32[8] slice(%elementwise.2), slice={[0:8]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  auto expected_scatter =
      m::Scatter(m::Slice(m::Constant()), m::Parameter(0), m::Parameter(1));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Abs(expected_scatter),
                                m::Exp(m::Abs(expected_scatter)))
                             .WithShape(F32, {8})));
}

TEST_F(ScatterSliceSimplifierTest, DiamondShape) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32_mul_F32 {
  %lhs.0 = f32[] parameter(0)
  %rhs.0 = f32[] parameter(2)
  %add.0 = f32[] add(%lhs.0, %rhs.0)
  %lhs.1 = f32[] parameter(1)
  %rhs.1 = f32[] parameter(3)
  %mul.1 = f32[] multiply(%lhs.1, %rhs.1)
  ROOT %tuple = (f32[], f32[]) tuple(%add.0, %mul.1)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates.0 = f32[4] parameter(1)
  %updates.1 = f32[4] parameter(2)
  %operands.0 = f32[9] constant(0)
  %operands.1 = f32[9] constant(0)
  %scatter = (f32[9], f32[9]) scatter(%operands.0, %operands.1, %indices, %updates.0, %updates.1), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32_mul_F32
  %gte.0 = f32[9] get-tuple-element(%scatter), index=0
  %gte.1 = f32[9] get-tuple-element(%scatter), index=1
  %consumer = f32[9] add(%gte.0, %gte.1)
  ROOT %slice = f32[8] slice(%consumer), slice={[0:8]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  auto expected_scatter =
      m::Scatter(m::Slice(m::Constant()), m::Slice(m::Constant()),
                 m::Parameter(0), m::Parameter(1), m::Parameter(2));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::GetTupleElement(expected_scatter),
                                m::GetTupleElement(expected_scatter))
                             .WithShape(F32, {8})));
}

TEST_F(ScatterSliceSimplifierTest, ElementwiseSelect) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module

%add_F32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}

ENTRY main {
  %indices = s32[4] parameter(0)
  %updates = f32[4] parameter(1)
  %operands = f32[9] constant(0)
  %scatter = f32[9] scatter(%operands, %indices, %updates), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%add_F32
  %pred_ = pred[9] parameter(2)
  %select = f32[9] select(%pred_, %scatter, %operands)
  ROOT %slice = f32[8] slice(%select), slice={[0:8]}
}
  )")
                    .value();
  ScatterSliceSimplifier test_pass;
  ASSERT_TRUE(RunHloPass(&test_pass, module.get()).value());
  auto expected_scatter =
      m::Scatter(m::Slice(m::Constant()), m::Parameter(0), m::Parameter(1));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Select(m::Slice(m::Parameter(2)), expected_scatter,
                                   m::Slice(m::Constant()))
                             .WithShape(F32, {8})));
}

}  // namespace
}  // namespace xla
