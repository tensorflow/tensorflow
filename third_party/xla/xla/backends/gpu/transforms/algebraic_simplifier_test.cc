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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuAlgebraicSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  se::CudaComputeCapability Ampere() {
    return se::CudaComputeCapability::Ampere();
  }
  se::CudaComputeCapability Hopper() {
    return se::CudaComputeCapability::Hopper();
  }
};

TEST_F(GpuAlgebraicSimplifierTest, SinkBroadcastOperandsOfChainedAdds) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  ASSERT_TRUE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::Broadcast(m::Add(m::Constant(), m::Constant())),
                  m::Convolution(m::Op(), m::Op()))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWhenDisabled) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={1}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(false);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkBroadcastOperandsOfChainedAddsWithoutConvolution) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      p = bf16[4, 4] parameter(0)
      const0 = bf16[4] constant({0, 0.25, 0.5, 0.75})
      bcast0 = bf16[4,4] broadcast(const0), dimensions={0}
      add0 = bf16[4,4] add(p, bcast0)
      const1 = bf16[4] constant({1, 1.25, 1.5, 1.75})
      bcast1 = bf16[4,4] broadcast(const1), dimensions={0}
      ROOT add1 = bf16[4,4] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(
    GpuAlgebraicSimplifierTest,
    DoNotSinkBroadcastOperandsOfChainedAddsWithMismatchedBroadcastDimensions) {
  const std::string& hlo_string = R"(
    HloModule m
    test {
      in = bf16[1,3,3,1] parameter(0)
      filter = bf16[2,2,1,1] constant({{{{1.1}}, {{2.1}}},
                                      {{{3.1}}, {{4.1}}}})
      conv = bf16[1,2,2,1] convolution(in, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      const0 = bf16[2] constant({0, 0.25})
      bcast0 = bf16[1,2,2,1] broadcast(const0), dimensions={1}
      add0 = bf16[1,2,2,1] add(conv, bcast0)
      const1 = bf16[2] constant({1, 1.25})
      bcast1 = bf16[1,2,2,1] broadcast(const1), dimensions={2}
      ROOT add1 = bf16[1,2,2,1] add(add0, bcast1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_sink_broadcast(true);
  EXPECT_FALSE(
      GpuAlgebraicSimplifier(options, se::CudaComputeCapability::Ampere())
          .Run(module.get())
          .value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X3_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x3
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(GpuAlgebraicSimplifierTest,
       DotToMultiplyRewriteWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,128] parameter(0)
      b = f32[128,128] parameter(1)
      ROOT dot = f32[128] dot(a, b),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1},
        algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
}

TEST_F(
    GpuAlgebraicSimplifierTest,
    DotToMultiplyRewriteForZeroContractingDimWith_BF16_BF16_F32_X6_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT dot = f32[] dot(a, b),
      algorithm=dot_bf16_bf16_f32_x6
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_TRUE(GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()).value());
  constexpr absl::string_view kPattern = R"(
    CHECK-COUNT-6: %[[partial_result:.*]] = bf16[] multiply
  )";
  ASSERT_OK_AND_ASSIGN(bool matched, RunFileCheck(m->ToString(), kPattern));
  EXPECT_TRUE(matched);
}

TEST_F(GpuAlgebraicSimplifierTest, SimplifyOneHotReduceSumToGather) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      idx = s32[32768] parameter(0)
      weights = f32[2048,512] parameter(1)
      bcast_idx = s32[32768,512] broadcast(idx), dimensions={0}
      iota = s32[32768,512] iota(), iota_dimension=1
      eq = pred[32768,512] compare(bcast_idx, iota), direction=EQ
      bcast_w = f32[1,2048,16,512] broadcast(weights), dimensions={1,3}
      reshape_w = f32[32768,512] reshape(bcast_w)
      zero = f32[] constant(0)
      bcast_zero = f32[32768,512] broadcast(zero), dimensions={}
      sel = f32[32768,512] select(eq, reshape_w, bcast_zero)
      ROOT red = f32[32768] reduce(sel, zero), dimensions={1}, to_apply=add_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_THAT(GpuAlgebraicSimplifier(options, Hopper()).Run(m.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Select(
          m::And(m::Compare(m::Parameter(0), m::Broadcast(m::Constant())),
                 m::Compare(m::Parameter(0), m::Broadcast(m::Constant()))),
          m::Gather(m::Reshape(m::Broadcast(m::Parameter(1))),
                    m::Concatenate(m::Iota(), m::Reshape(m::Parameter(0)))),
          m::Broadcast(m::Constant()))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       SimplifyOneHotReduceSumPeelsBroadcastOnTrue) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      idx = s32[16,2048] parameter(0)
      weights = f32[2048,512] parameter(1)
      bcast_idx = s32[16,2048,512] broadcast(idx), dimensions={0,1}
      iota = s32[16,2048,512] iota(), iota_dimension=2
      eq = pred[16,2048,512] compare(bcast_idx, iota), direction=EQ
      bcast_w = f32[16,2048,512] broadcast(weights), dimensions={1,2}
      zero = f32[] constant(0)
      bcast_zero = f32[16,2048,512] broadcast(zero), dimensions={}
      sel = f32[16,2048,512] select(eq, bcast_w, bcast_zero)
      ROOT red = f32[16,2048] reduce(sel, zero), dimensions={2}, to_apply=add_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  ASSERT_THAT(GpuAlgebraicSimplifier(options, Hopper()).Run(m.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Select(
          m::And(m::Compare(m::Parameter(0), m::Broadcast(m::Constant())),
                 m::Compare(m::Parameter(0), m::Broadcast(m::Constant()))),
          m::Gather(m::Parameter(1),
                    m::Concatenate(m::Iota(), m::Reshape(m::Parameter(0)))),
          m::Broadcast(m::Constant()))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       SimplifyOneHotReduceSumPreservesOutOfBoundsSemantics) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      idx = s32[2,3] parameter(0)
      weights = f32[2,16] parameter(1)
      bcast_idx = s32[2,3,16] broadcast(idx), dimensions={0,1}
      iota = s32[2,3,16] iota(), iota_dimension=2
      eq = pred[2,3,16] compare(bcast_idx, iota), direction=EQ
      bcast_w = f32[2,3,16] broadcast(weights), dimensions={0,2}
      zero = f32[] constant(0)
      bcast_zero = f32[2,3,16] broadcast(zero), dimensions={}
      sel = f32[2,3,16] select(eq, bcast_w, bcast_zero)
      ROOT red = f32[2,3] reduce(sel, zero), dimensions={2}, to_apply=add_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m_before, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_OK_AND_ASSIGN(auto m_after, ParseAndReturnVerifiedModule(kModuleStr));

  // Include negative, valid (0..15), and out-of-bounds (16, 100) indices.
  Literal idx_lit = LiteralUtil::CreateR2<int32_t>({{-1, 0, 15}, {16, 5, 100}});
  Literal weights_lit = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
       {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f,
        110.0f, 120.0f, 130.0f, 140.0f, 150.0f, 160.0f}});

  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       evaluator.Evaluate(*m_before, {&idx_lit, &weights_lit}));

  AlgebraicSimplifierOptions options;
  ASSERT_THAT(GpuAlgebraicSimplifier(options, Hopper()).Run(m_after.get()),
              absl_testing::IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(Literal actual,
                       evaluator.Evaluate(*m_after, {&idx_lit, &weights_lit}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

TEST_F(GpuAlgebraicSimplifierTest,
       SimplifyOneHotReduceSumPeelsNarrowConvertAndBroadcastOnTrue) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      idx = s32[2,3] parameter(0)
      weights = bf16[3,8] parameter(1)
      bcast_idx = s32[2,3,8] broadcast(idx), dimensions={0,1}
      iota = s32[2,3,8] iota(), iota_dimension=2
      eq = pred[2,3,8] compare(bcast_idx, iota), direction=EQ
      bcast_w = bf16[2,3,8] broadcast(weights), dimensions={1,2}
      conv_w = f32[2,3,8] convert(bcast_w)
      zero = f32[] constant(0)
      bcast_zero = f32[2,3,8] broadcast(zero), dimensions={}
      sel = f32[2,3,8] select(eq, conv_w, bcast_zero)
      ROOT red = f32[2,3] reduce(sel, zero), dimensions={2}, to_apply=add_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m_before, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_OK_AND_ASSIGN(auto m_after, ParseAndReturnVerifiedModule(kModuleStr));

  Literal idx_lit = LiteralUtil::CreateR2<int32_t>({{-1, 0, 7}, {8, 5, 2}});
  ASSERT_OK_AND_ASSIGN(
      Literal weights_lit,
      LiteralUtil::CreateR2<float>(
          {{1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f},
           {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f},
           {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}})
          .Convert(BF16));

  ASSERT_OK_AND_ASSIGN(
      Literal expected,
      HloEvaluator().Evaluate(*m_before, {&idx_lit, &weights_lit}));

  AlgebraicSimplifierOptions options;
  ASSERT_THAT(GpuAlgebraicSimplifier(options, Hopper()).Run(m_after.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      m_after->entry_computation()->root_instruction(),
      GmockMatch(m::Select(
          m::And(m::Compare(m::Parameter(0), m::Broadcast(m::Constant())),
                 m::Compare(m::Parameter(0), m::Broadcast(m::Constant()))),
          m::Convert(m::Gather(
              m::Parameter(1),
              m::Concatenate(m::Iota(), m::Reshape(m::Parameter(0))))),
          m::Broadcast(m::Constant()))));

  ASSERT_OK_AND_ASSIGN(Literal actual, HloEvaluator().Evaluate(
                                           *m_after, {&idx_lit, &weights_lit}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

TEST_F(GpuAlgebraicSimplifierTest, SinkReduceThroughSlice) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[8,256,1536]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[256,1536]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      ROOT s = f32[256,1024]{1,0} slice(r), slice={[0:256], [0:1024]}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                             .WithShape(m::Shape().WithDims({256, 1024}))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       SinkReduceAndConvertThroughPartitioningSlices) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[8,256,1536]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[256,1536]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      cvt = bf16[256,1536]{1,0} convert(r)
      slice0 = bf16[256,1024]{1,0} slice(cvt), slice={[0:256], [0:1024]}
      slice1 = bf16[256,512]{1,0} slice(cvt), slice={[0:256], [1024:1536]}
      ROOT t = (bf16[256,1024]{1,0}, bf16[256,512]{1,0}) tuple(slice0, slice1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Convert(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                         .WithShape(m::Shape().WithDims({256, 1024}))),
          m::Convert(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                         .WithShape(m::Shape().WithDims({256, 512}))))));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkReduceThroughOverlappingSlicesExceedingSize) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[8,256,1536]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[256,1536]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      cvt = bf16[256,1536]{1,0} convert(r)
      slice0 = bf16[256,1024]{1,0} slice(cvt), slice={[0:256], [0:1024]}
      slice1 = bf16[256,1024]{1,0} slice(cvt), slice={[0:256], [512:1536]}
      ROOT t = (bf16[256,1024]{1,0}, bf16[256,1024]{1,0}) tuple(slice0, slice1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(GpuAlgebraicSimplifierTest,
       SinkReduceThroughConvertAndBitcastPartitioningSlices) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[3,16,48]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[16,48]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      cvt = bf16[16,48]{1,0} convert(r)
      bc = bf16[4,4,12,4]{3,2,1,0} bitcast(cvt)
      slice_q = bf16[4,4,8,4]{3,2,1,0} slice(bc),
          slice={[0:4], [0:4], [0:8], [0:4]}
      slice_k = bf16[4,4,2,4]{3,2,1,0} slice(bc),
          slice={[0:4], [0:4], [8:10], [0:4]}
      slice_v = bf16[4,4,2,4]{3,2,1,0} slice(bc),
          slice={[0:4], [0:4], [10:12], [0:4]}
      ROOT t = (bf16[4,4,8,4]{3,2,1,0}, bf16[4,4,2,4]{3,2,1,0},
                bf16[4,4,2,4]{3,2,1,0}) tuple(slice_q, slice_k, slice_v)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m_before, ParseAndReturnVerifiedModule(kModuleStr));
  Literal input =
      LiteralUtil::CreateFullWithDescendingLayout<float>({3, 16, 48}, 0.0f);
  for (int64_t i = 0; i < input.element_count(); ++i) {
    input.data<float>()[i] = static_cast<float>((i % 13) - 6) * 0.25f;
  }
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       HloEvaluator().Evaluate(*m_before, {&input}));

  ASSERT_OK_AND_ASSIGN(auto m_after, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GpuAlgebraicSimplifier(options, Ampere()).Run(m_after.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      m_after->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Bitcast(
              m::Convert(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                             .WithShape(m::Shape().WithDims({16, 32})))),
          m::Bitcast(
              m::Convert(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                             .WithShape(m::Shape().WithDims({16, 8})))),
          m::Bitcast(
              m::Convert(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                             .WithShape(m::Shape().WithDims({16, 8})))))));

  ASSERT_OK_AND_ASSIGN(Literal actual,
                       HloEvaluator().Evaluate(*m_after, {&input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

TEST_F(GpuAlgebraicSimplifierTest,
       DoNotSinkReduceThroughNonContiguousBitcastSlice) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[3,16,48]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[16,48]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      bc = f32[4,4,12,4]{3,2,1,0} bitcast(r)
      ROOT s = f32[4,4,8,2]{3,2,1,0} slice(bc),
          slice={[0:4], [0:4], [0:8], [0:2]}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(GpuAlgebraicSimplifierTest, SinkReduceThroughReshapeLayoutInsensitive) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[4,16,48] parameter(0)
      c0 = f32[] constant(0)
      r = f32[16,48] reduce(p0, c0), dimensions={0}, to_apply=add
      rs = f32[4,4,12,4] reshape(r)
      slice0 = f32[4,4,8,4] slice(rs), slice={[0:4], [0:4], [0:8], [0:4]}
      slice1 = f32[4,4,4,4] slice(rs), slice={[0:4], [0:4], [8:12], [0:4]}
      ROOT t = (f32[4,4,8,4], f32[4,4,4,4]) tuple(slice0, slice1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(false);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Reshape(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                                 .WithShape(m::Shape().WithDims({16, 32}))),
                  m::Reshape(m::Reduce(m::Slice(m::Parameter(0)), m::Constant())
                                 .WithShape(m::Shape().WithDims({16, 16}))))));
}

TEST_F(GpuAlgebraicSimplifierTest, DoNotSinkReduceWithDeadPassthroughUser) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[8,256,1536]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[256,1536]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add
      dead_cvt = bf16[256,1536]{1,0} convert(r)
      ROOT s = f32[256,1024]{1,0} slice(r), slice={[0:256], [0:1024]}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(GpuAlgebraicSimplifierTest, DoNotSinkReduceExceedingMaxLeafSlices) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule test
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY main {
      p0 = f32[8,17]{1,0} parameter(0)
      c0 = f32[] constant(0)
      r = f32[17]{0} reduce(p0, c0), dimensions={0}, to_apply=add
      slice0 = f32[1]{0} slice(r), slice={[0:1]}
      slice1 = f32[1]{0} slice(r), slice={[1:2]}
      slice2 = f32[1]{0} slice(r), slice={[2:3]}
      slice3 = f32[1]{0} slice(r), slice={[3:4]}
      slice4 = f32[1]{0} slice(r), slice={[4:5]}
      slice5 = f32[1]{0} slice(r), slice={[5:6]}
      slice6 = f32[1]{0} slice(r), slice={[6:7]}
      slice7 = f32[1]{0} slice(r), slice={[7:8]}
      slice8 = f32[1]{0} slice(r), slice={[8:9]}
      slice9 = f32[1]{0} slice(r), slice={[9:10]}
      slice10 = f32[1]{0} slice(r), slice={[10:11]}
      slice11 = f32[1]{0} slice(r), slice={[11:12]}
      slice12 = f32[1]{0} slice(r), slice={[12:13]}
      slice13 = f32[1]{0} slice(r), slice={[13:14]}
      slice14 = f32[1]{0} slice(r), slice={[14:15]}
      slice15 = f32[1]{0} slice(r), slice={[15:16]}
      slice16 = f32[1]{0} slice(r), slice={[16:17]}
      ROOT t = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
                f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
                f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0},
                f32[1]{0}, f32[1]{0})
          tuple(slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7,
                slice8, slice9, slice10, slice11, slice12, slice13, slice14,
                slice15, slice16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       GpuAlgebraicSimplifier(options, Ampere()).Run(m.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla::gpu
