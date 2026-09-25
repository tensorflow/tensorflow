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

#include "xla/hlo/transforms/expanders/shuffle_expander.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ShuffleExpanderTest = HloHardwareIndependentTestBase;
namespace m = testing::opcode_matchers;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

TEST_F(ShuffleExpanderTest, ExpandsRotate1D) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT shuffle = f32[4]{0} shuffle(p0), dimensions={0}, mode=rotate, shifts={1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* p0 = module->entry_computation()->parameter_instruction(0);
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Concatenate(m::Slice(m::Parameter(0)),
                                   m::Slice(m::Parameter(0))));
  EXPECT_EQ(root->concatenate_dimension(), 0);

  auto* slice0 = Cast<HloSliceInstruction>(root->operand(0));
  auto* slice1 = Cast<HloSliceInstruction>(root->operand(1));

  EXPECT_EQ(slice0->operand(0), p0);
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(1));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(4));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1));

  EXPECT_EQ(slice1->operand(0), p0);
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1));
}

TEST_F(ShuffleExpanderTest, ExpandsRotate2D) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  ROOT shuffle = f32[2,3]{1,0} shuffle(p0), dimensions={0,1}, mode=rotate, shifts={1,2}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ShuffleExpander expander;

  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  auto* p0 = module->entry_computation()->parameter_instruction(0);
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Concatenate(
                        m::Slice(m::Concatenate(m::Slice(m::Parameter(0)),
                                                m::Slice(m::Parameter(0)))),
                        m::Slice(m::Concatenate(m::Slice(m::Parameter(0)),
                                                m::Slice(m::Parameter(0))))));
  EXPECT_EQ(root->concatenate_dimension(), 1);

  // The rotation of dimension 0 by 1.
  auto* dim0_rotate =
      Cast<HloConcatenateInstruction>(root->operand(0)->operand(0));
  EXPECT_EQ(root->operand(1)->operand(0), dim0_rotate);
  EXPECT_EQ(dim0_rotate->concatenate_dimension(), 0);

  auto* dim0_slice0 = Cast<HloSliceInstruction>(dim0_rotate->operand(0));
  auto* dim0_slice1 = Cast<HloSliceInstruction>(dim0_rotate->operand(1));

  EXPECT_EQ(dim0_slice0->operand(0), p0);
  EXPECT_THAT(dim0_slice0->slice_starts(), ElementsAre(1, 0));
  EXPECT_THAT(dim0_slice0->slice_limits(), ElementsAre(2, 3));
  EXPECT_THAT(dim0_slice0->slice_strides(), ElementsAre(1, 1));

  EXPECT_EQ(dim0_slice1->operand(0), p0);
  EXPECT_THAT(dim0_slice1->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(dim0_slice1->slice_limits(), ElementsAre(1, 3));
  EXPECT_THAT(dim0_slice1->slice_strides(), ElementsAre(1, 1));

  // The rotation of dimension 1 by 2.
  auto* dim1_slice0 = Cast<HloSliceInstruction>(root->operand(0));
  auto* dim1_slice1 = Cast<HloSliceInstruction>(root->operand(1));

  EXPECT_THAT(dim1_slice0->slice_starts(), ElementsAre(0, 2));
  EXPECT_THAT(dim1_slice0->slice_limits(), ElementsAre(2, 3));
  EXPECT_THAT(dim1_slice0->slice_strides(), ElementsAre(1, 1));

  EXPECT_THAT(dim1_slice1->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(dim1_slice1->slice_limits(), ElementsAre(2, 2));
  EXPECT_THAT(dim1_slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(ShuffleExpanderTest, SimplifiesZeroShiftRotate) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT shuffle = f32[4]{0} shuffle(p0), dimensions={0}, mode=rotate, shifts={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

// Expects `gather` to take one element of its operand per element of its
// result, indexing every dimension of the operand with the components of the
// innermost dimension of its start indices.
void ExpectElementwiseGather(const HloInstruction* gather) {
  const int64_t rank = gather->operand(0)->shape().dimensions().size();
  std::vector<int64_t> all_dimensions(rank);
  absl::c_iota(all_dimensions, 0);
  auto* elementwise_gather = Cast<HloGatherInstruction>(gather);
  const GatherDimensionNumbers& dimension_numbers =
      elementwise_gather->gather_dimension_numbers();
  EXPECT_THAT(dimension_numbers.offset_dims(), IsEmpty());
  EXPECT_THAT(dimension_numbers.collapsed_slice_dims(),
              ElementsAreArray(all_dimensions));
  EXPECT_THAT(dimension_numbers.start_index_map(),
              ElementsAreArray(all_dimensions));
  EXPECT_EQ(dimension_numbers.index_vector_dim(), rank);
  EXPECT_THAT(elementwise_gather->gather_slice_sizes(), Each(1));
  EXPECT_FALSE(elementwise_gather->indices_are_sorted());
  EXPECT_EQ(gather->shape(), gather->operand(0)->shape());
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteOfOneDimension) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[3,2]{1,0} parameter(0)
  ROOT shuffle = f32[3,2]{1,0} shuffle(p0), dimensions={0}, mode=permute, indices=s32[3,1]{1,0} { {2}, {0}, {1} }
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // The indices share their coordinates along dimension 1, which the broadcast
  // materializes once a reshape drops that dimension, and dimension 1 keeps its
  // own coordinate via an iota.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              m::Gather(m::Parameter(0),
                        m::Concatenate(m::Broadcast(m::Reshape(m::Constant())),
                                       m::Iota())));
  ExpectElementwiseGather(root);

  auto* concatenate = root->operand(1);
  EXPECT_THAT(concatenate, m::Shape("s32[3,2,2]"));
  EXPECT_EQ(concatenate->concatenate_dimension(), 2);

  auto* broadcast = concatenate->operand(0);
  EXPECT_THAT(broadcast, m::Shape("s32[3,2,1]"));
  EXPECT_THAT(broadcast->dimensions(), ElementsAre(0));
  EXPECT_THAT(broadcast->operand(0), m::Shape("s32[3]"));
  EXPECT_EQ(broadcast->operand(0)->operand(0)->literal(),
            LiteralUtil::CreateR2<int32_t>({{2}, {0}, {1}}));

  auto* iota = concatenate->operand(1);
  EXPECT_THAT(iota, m::Shape("s32[3,2,1]"));
  EXPECT_EQ(Cast<HloIotaInstruction>(iota)->iota_dimension(), 1);
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteOfEveryDimensionInOrder) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[3,2]{1,0} parameter(0)
  ROOT shuffle = f32[3,2]{1,0} shuffle(p0), dimensions={0,1}, mode=permute, indices=s32[3,2,2]{2,1,0} { { {2,0}, {0,1} }, { {0,0}, {2,1} }, { {1,0}, {1,1} } }
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // The indices already hold the gather's index vectors, so no broadcast,
  // iota, slice or concatenate is needed.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Gather(m::Parameter(0), m::Constant()));
  ExpectElementwiseGather(root);
  EXPECT_EQ(root->operand(1)->literal(),
            LiteralUtil::CreateR3<int32_t>(
                {{{2, 0}, {0, 1}}, {{0, 0}, {2, 1}}, {{1, 0}, {1, 1}}}));
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteOfRank1) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT shuffle = f32[4]{0} shuffle(p0), dimensions={0}, mode=permute, indices=s32[4]{0} {3, 1, 0, 2}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // The only shuffled dimension is also every dimension, so the broadcast that
  // adds the coordinate dimension is all that the indices need.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Gather(m::Parameter(0), m::Broadcast(m::Constant())));
  ExpectElementwiseGather(root);

  auto* broadcast = root->operand(1);
  EXPECT_THAT(broadcast, m::Shape("s32[4,1]"));
  EXPECT_THAT(broadcast->dimensions(), ElementsAre(0));
  EXPECT_EQ(broadcast->operand(0)->literal(),
            LiteralUtil::CreateR1<int32_t>({3, 1, 0, 2}));
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteOfEveryDimensionOutOfOrder) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[3,2]{1,0} parameter(0)
  ROOT shuffle = f32[3,2]{1,0} shuffle(p0), dimensions={1,0}, mode=permute, indices=s32[3,2,2]{2,1,0} { { {0,2}, {1,0} }, { {0,0}, {1,2} }, { {0,1}, {1,1} } }
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // A coordinate of the indices belongs to the dimension that `dimensions`
  // lists at its position, while the gather indexes the dimensions in order,
  // so the coordinates are sliced apart and concatenated back in the order of
  // the operand: the slice of coordinate 1 comes first because it holds the
  // coordinates of dimension 0.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Gather(m::Parameter(0),
                              m::Concatenate(m::Slice(m::Constant()),
                                             m::Slice(m::Constant()))));
  ExpectElementwiseGather(root);

  auto* concatenate = root->operand(1);
  EXPECT_THAT(concatenate, m::Shape("s32[3,2,2]"));
  EXPECT_EQ(concatenate->concatenate_dimension(), 2);

  auto* slice0 = Cast<HloSliceInstruction>(concatenate->operand(0));
  auto* slice1 = Cast<HloSliceInstruction>(concatenate->operand(1));
  auto* constant = slice0->operand(0);
  EXPECT_EQ(slice1->operand(0), constant);
  EXPECT_EQ(constant->literal(),
            LiteralUtil::CreateR3<int32_t>(
                {{{0, 2}, {1, 0}}, {{0, 0}, {1, 2}}, {{0, 1}, {1, 1}}}));

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 1));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(3, 2, 2));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1));

  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0, 0, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(3, 2, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1));
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteOfNonAdjacentDimensions) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[2,2,2]{2,1,0} parameter(0)
  ROOT shuffle = f32[2,2,2]{2,1,0} shuffle(p0), dimensions={0,2}, mode=permute, indices=s32[2,1,2,2]{3,2,1,0} { { { {1,1}, {1,0} } }, { { {0,1}, {0,0} } } }
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // The indices share their coordinates along the dimension in between, which
  // a reshape drops and the broadcast materializes, and the iota of the
  // dimension that the shuffle leaves in place is concatenated between the two
  // sliced coordinates.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, m::Gather(m::Parameter(0),
                      m::Concatenate(
                          m::Slice(m::Broadcast(m::Reshape(m::Constant()))),
                          m::Iota(),
                          m::Slice(m::Broadcast(m::Reshape(m::Constant()))))));
  ExpectElementwiseGather(root);

  auto* concatenate = root->operand(1);
  EXPECT_THAT(concatenate, m::Shape("s32[2,2,2,3]"));
  EXPECT_EQ(concatenate->concatenate_dimension(), 3);

  auto* slice0 = Cast<HloSliceInstruction>(concatenate->operand(0));
  auto* iota = concatenate->operand(1);
  auto* slice1 = Cast<HloSliceInstruction>(concatenate->operand(2));

  auto* broadcast = slice0->operand(0);
  EXPECT_EQ(slice1->operand(0), broadcast);
  EXPECT_THAT(broadcast, m::Shape("s32[2,2,2,2]"));
  EXPECT_THAT(broadcast->dimensions(), ElementsAre(0, 2, 3));
  EXPECT_THAT(broadcast->operand(0), m::Shape("s32[2,2,2]"));
  EXPECT_EQ(
      broadcast->operand(0)->operand(0)->literal(),
      LiteralUtil::CreateR4<int32_t>({{{{1, 1}, {1, 0}}}, {{{0, 1}, {0, 0}}}}));

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 2, 2, 1));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1, 1));

  EXPECT_THAT(iota, m::Shape("s32[2,2,2,1]"));
  EXPECT_EQ(Cast<HloIotaInstruction>(iota)->iota_dimension(), 1);

  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0, 0, 0, 1));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(2, 2, 2, 2));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1, 1));
}

TEST_F(ShuffleExpanderTest, ExpandsPermuteWithNarrowIndices) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[3,2]{1,0} parameter(0)
  ROOT shuffle = f32[3,2]{1,0} shuffle(p0), dimensions={0,1}, mode=permute, indices=s8[3,2,2]{2,1,0} { { {2,0}, {0,1} }, { {0,0}, {2,1} }, { {1,0}, {1,1} } }
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  // The coordinates hold every dimension size, so indices of a narrower type
  // are converted up to s32.
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Gather(m::Parameter(0), m::Convert(m::Constant())));
  ExpectElementwiseGather(root);

  auto* convert = root->operand(1);
  EXPECT_THAT(convert, m::Shape("s32[3,2,2]"));
  EXPECT_EQ(convert->operand(0)->literal(),
            LiteralUtil::CreateR3<int8_t>(
                {{{2, 0}, {0, 1}}, {{0, 0}, {2, 1}}, {{1, 0}, {1, 1}}}));
}

}  // namespace
}  // namespace xla
