/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/cudnn_simplify_padding.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/literal.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class CudnnSimplifyPaddingTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunJustThisPass(HloModule* module) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        RunHloPass(CudnnSimplifyPadding(), module));
    VLOG(1) << "after simplify_padding:\n" << module->ToString();

    // I know the name says "just this pass", but you really want algsimp too,
    // otherwise the resulting patterns are ugly/hard to match.
    TF_RETURN_IF_ERROR(RunHloPass(HloPassFix<AlgebraicSimplifier>(
                                      AlgebraicSimplifierOptions()),
                                  module)
                           .status());
    return changed;
  }
};

void ExpectOnlyPadsOneDim(int64_t dim, int64_t padding_high,
                          const PaddingConfig& p) {
  SCOPED_TRACE(p.DebugString());
  for (int i = 0; i < p.dimensions_size(); ++i) {
    SCOPED_TRACE(absl::StrCat("dimension ", i));
    EXPECT_EQ(p.dimensions(i).edge_padding_low(), 0);
    if (i == dim) {
      EXPECT_EQ(p.dimensions(i).edge_padding_high(), padding_high);
    } else {
      EXPECT_EQ(p.dimensions(i).edge_padding_high(), 0);
    }
  }
}

template <typename NativeT>
void SetConstantValue(
    HloInstruction* instr,
    absl::FunctionRef<NativeT(absl::Span<const int64_t>, NativeT)> value_fn) {
  Literal new_literal = instr->literal().Clone();
  new_literal.MutableEachCell<int8_t>(value_fn);
  EXPECT_OK(instr->parent()->ReplaceWithNewInstruction(
      instr, HloInstruction::CreateConstant(std::move(new_literal))));
}

TEST_F(CudnnSimplifyPaddingTest, PaddedWeights) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  const HloInstruction* pad = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Pad(&pad, m::GetTupleElement(m::CustomCall(), 0),
                                m::ConstantScalar(0))));

  ExpectOnlyPadsOneDim(/*dim=*/3, /*padding_high=*/1, pad->padding_config());
}

// This is similar to PaddedWeights, except the only 3 elements of the weights
// are padded to 0 while we slice off 4 elements from the output features. As a
// result, not all of the sliced elements are 0, and we can't merge the slice
// into the pad that follows.
TEST_F(CudnnSimplifyPaddingTest, PaddedWeightsNotPaddedEnough) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_3
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, PaddedConstantWeight) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(0),
          s8[3,3,10,10] constant({...})
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  // Set the constant's value.  (The HLO text above sets it to all 0s.)
  {
    HloInstruction* weights = nullptr;
    ASSERT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(m::Pad(m::Slice(m::GetTupleElement(m::CustomCall(
                                      m::Op(), m::Constant(&weights)))),
                                  m::Op())));
    SetConstantValue<int8_t>(
        weights, [](absl::Span<const int64_t> dims, int8_t old_val) -> int8_t {
          if (dims[3] < 6) {
            return 1;
          }
          return 0;
        });
  }

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  const HloInstruction* pad = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Pad(&pad, m::GetTupleElement(m::CustomCall(), 0),
                                m::ConstantScalar(0))));

  ExpectOnlyPadsOneDim(/*dim=*/3, /*padding_high=*/1, pad->padding_config());
}

TEST_F(CudnnSimplifyPaddingTest, PaddedConstantWeightIsNotLargeEnough) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(0),
          s8[3,3,10,10] constant({...})
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  // Set the constant's value.  (The HLO text above sets it to all 0s.)
  {
    HloInstruction* weights = nullptr;
    ASSERT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(m::Pad(m::Slice(m::GetTupleElement(m::CustomCall(
                                      m::Op(), m::Constant(&weights)))),
                                  m::Op())));
    SetConstantValue<int8_t>(
        weights, [](absl::Span<const int64_t> dims, int8_t old_val) -> int8_t {
          // The sixth feature dimension (i.e. index 5) is only partially 0.
          if (dims[3] < 5 /*|| (dims[3] == 5 && dims[2] > 1)*/) return 0;
          return 1;
        });
  }

  // Some of the value sliced off are not 0, so we can't merge the slice into
  // the pad.
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, SliceDoesntStartAtBeginning) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,9,10,6] slice(conv_result), slice={[0:10], [1:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, SliceDoesntStartAtBeginningOfFeatureDim) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,5] slice(conv_result), slice={[0:10], [0:10], [0:10], [1:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, SliceHasStride) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,3] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6:2]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, PadAddsInteriorPadding) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_5_1
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, SliceMoreElementsThanPad) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule TestModule

    ENTRY TestComputation {
      weights = pad(s8[3,3,10,10] parameter(0), s8[] constant(0)), padding=0_0x0_0x0_0x0_4
      conv = (s8[10,10,10,10], u8[0]) custom-call(
          s8[10,10,10,10] parameter(1),
          weights
        ), window={size=3x3}, dim_labels=b01f_01io->b01f,
        custom_call_target="__cudnn$convForward"
      conv_result = get-tuple-element(conv), index=0
      slice = s8[10,10,10,6] slice(conv_result), slice={[0:10], [0:10], [0:10], [0:6]}
      ROOT pad = pad(slice, s8[] constant(0)), padding=0_0x0_0x0_0x0_2
    }
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  const HloInstruction* slice = nullptr;
  // The pass creates a pad with negative padding; this is simplified by algsimp
  // into a slice.
  ASSERT_THAT(root, GmockMatch(m::Slice(
                        &slice, m::GetTupleElement(m::CustomCall(), 0))));
  for (int64_t i = 0; i < slice->shape().dimensions().size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(slice->slice_starts(i), 0);
    EXPECT_EQ(slice->slice_strides(i), 1);
    if (i != 3) {
      EXPECT_EQ(slice->slice_limits(i), 10);
    } else {
      EXPECT_EQ(slice->slice_limits(i), 8);
    }
  }
}

TEST_F(CudnnSimplifyPaddingTest, NoChangeOnNonTrivialConstants) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule jit_outer

ENTRY main.26 {
  reshape.2 = f32[1,3,3,12]{3,2,1,0} parameter(0)
  constant.1 = f32[3,3,1,12]{3,2,1,0} constant({ {
    { /*i1=0*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
    { /*i1=2*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } })
  cudnn-conv = (f32[1,5,5,12]{3,2,1,0}, u8[0]{0}) custom-call(reshape.2, constant.1), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01io->b01f, feature_group_count=12, custom_call_target="__cudnn$convForward"
  get-tuple-element = f32[1,5,5,12]{3,2,1,0} get-tuple-element(cudnn-conv), index=0
  slice.2 = f32[1,5,1,12]{3,2,1,0} slice(get-tuple-element), slice={[0:1], [0:5], [0:1], [0:12]}
  constant.0 = f32[] constant(0)
  ROOT pad.1 = f32[1,5,3,12]{3,2,1,0} pad(slice.2, constant.0), padding=0_0x0_0x2_0x0_0
}
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, NoChangeOnComplexSlices) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule jit_outer

ENTRY main.26 {
  reshape.2 = f32[1,3,3,12]{3,2,1,0} parameter(0)
  constant.1 = f32[3,3,1,12]{3,2,1,0} constant({ {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } })
  cudnn-conv = (f32[1,5,5,12]{3,2,1,0}, u8[0]{0}) custom-call(reshape.2, constant.1), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01io->b01f, feature_group_count=12, custom_call_target="__cudnn$convForward"
  get-tuple-element = f32[1,5,5,12]{3,2,1,0} get-tuple-element(cudnn-conv), index=0
  slice.2 = f32[1,5,5,4]{3,2,1,0} slice(get-tuple-element), slice={[0:1], [0:5], [0:5], [2:6]}
  constant.0 = f32[] constant(0)
  ROOT pad.1 = f32[1,5,5,12]{3,2,1,0} pad(slice.2, constant.0), padding=0_0x0_0x0_0x0_8
}
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnSimplifyPaddingTest, ScanOrderFeatureDimLast) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule jit_outer

ENTRY main.26 {
  reshape.2 = f32[1,3,3,12]{3,2,1,0} parameter(0)
  constant.1 = f32[3,3,1,12]{3,2,1,0} constant({ {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }
  }, {
    { /*i1=0*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=1*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    { /*i1=2*/ { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } })
  cudnn-conv = (f32[1,5,5,12]{3,2,1,0}, u8[0]{0}) custom-call(reshape.2, constant.1), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01io->b01f, feature_group_count=12, custom_call_target="__cudnn$convForward"
  get-tuple-element = f32[1,5,5,12]{3,2,1,0} get-tuple-element(cudnn-conv), index=0
  slice.2 = f32[1,5,5,6]{3,2,1,0} slice(get-tuple-element), slice={[0:1], [0:5], [0:5], [0:6]}
  constant.0 = f32[] constant(0)
  ROOT pad.1 = f32[1,5,5,12]{3,2,1,0} pad(slice.2, constant.0), padding=0_0x0_0x0_0x0_6
}
  )")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunJustThisPass(module.get()));
  EXPECT_FALSE(changed);
}

}  // anonymous namespace
}  // namespace xla::gpu
