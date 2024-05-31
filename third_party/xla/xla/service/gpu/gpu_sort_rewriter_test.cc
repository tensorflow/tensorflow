/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_sort_rewriter.h"

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuSortRewriterTest : public HloTestBase {
 public:
  bool RunPass(HloModule* module) {
    return GpuSortRewriter().Run(module).value();
  }

  void ExpectDirection(const HloInstruction* instruction, bool descending) {
    auto config = instruction->backend_config<xla::SortOptions>();
    EXPECT_EQ(config->descending(), descending);
  }
};

// Basic sort: ascending.
TEST_F(GpuSortRewriterTest, SortKeysLessThan) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100000] parameter(0)
  ROOT %sort = f32[100000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: descending.
TEST_F(GpuSortRewriterTest, SortKeysGreaterThan) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[100000] parameter(0)
  ROOT %sort = f32[100000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/true);
}

// Comparer swaps the parameter order -> direction is reversed.
TEST_F(GpuSortRewriterTest, SortKeysGreaterThanSwapped) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(1)
  %rhs = f32[] parameter(0)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[100000] parameter(0)
  ROOT %sort = f32[100000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Sort a pair of tensors, keys go first.
TEST_F(GpuSortRewriterTest, SortPairs) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_key = u32[] parameter(0)
  %rhs_key = u32[] parameter(1)
  %lhs_value = f32[] parameter(2)
  %rhs_value = f32[] parameter(3)
  ROOT %lt = pred[] compare(%lhs_key, %rhs_key), direction=LT
}

ENTRY %main {
  %input_keys = u32[100000] parameter(0)
  %input_values = f32[100000] parameter(1)
  ROOT %sort = (u32[100000], f32[100000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

// Sort a pair of tensors, keys go last.
TEST_F(GpuSortRewriterTest, SortPairsSwapped) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(1)
  %lhs_key = u32[] parameter(2)
  %rhs_key = u32[] parameter(3)
  ROOT %lt = pred[] compare(%lhs_key, %rhs_key), direction=LT
}

ENTRY %main {
  %input_values = f32[100000] parameter(0)
  %input_keys = u32[100000] parameter(1)
  ROOT %sort = (f32[100000], u32[100000]) sort(%input_values, %input_keys),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 1),
                                  m::GetTupleElement(m::CustomCall(), 0))));
}

// CUB sort doesn't support more than two tensors.
TEST_F(GpuSortRewriterTest, NoRewriteManyTensors) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  %unused1 = f64[] parameter(2)
  %unused2 = f64[] parameter(3)
  %unused3 = u64[] parameter(4)
  %unused4 = u64[] parameter(5)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input1 = f32[100000] parameter(0)
  %input2 = f64[100000] parameter(1)
  %input3 = u64[100000] parameter(2)
  ROOT %sort = (f32[100000], f64[100000], u64[100000]) sort(%input1, %input2, %input3),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Only 1D shapes are supported.
TEST_F(GpuSortRewriterTest, NoRewriteNonMinorSortDimension) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100000,4] parameter(0)
  ROOT %sort = f32[100000,4] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Kernels are compiled for a subset of types.
TEST_F(GpuSortRewriterTest, NoRewriteUnsupportedType) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = pred[] parameter(0)
  %rhs = pred[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = pred[100000] parameter(0)
  ROOT %sort = pred[100000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Comparer must be a simple function.
TEST_F(GpuSortRewriterTest, NoRewriteComplexComparer) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %lhs_scaled = f32[] multiply(%lhs, f32[] constant(2))
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs_scaled, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100000] parameter(0)
  ROOT %sort = f32[100000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Comparer must use adjacent input values.
TEST_F(GpuSortRewriterTest, NoRewriteMixedKeysValues) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_key = u32[] parameter(0)
  %rhs_key = u32[] parameter(1)
  %lhs_value = u32[] parameter(2)
  %rhs_value = u32[] parameter(3)
  ROOT %mixed = pred[] compare(%rhs_key, %lhs_value), direction=LT
}

ENTRY %main {
  %input_keys = u32[100000] parameter(0)
  %input_values = u32[100000] parameter(1)
  ROOT %sort = (u32[100000], u32[100000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Small shapes do not see improvement from CUB sort.
TEST_F(GpuSortRewriterTest, NoRewriteSmallSize) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunPass(module.get()));
}

// Basic sort: with batch dimension.
TEST_F(GpuSortRewriterTest, SortWithBatchDim) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100,1000] parameter(0)
  ROOT %sort = f32[100,1000] sort(%input), dimensions={1}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: with multiple batch dimensions.
TEST_F(GpuSortRewriterTest, SortWithMultipleBatchDims) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[10,10,1000] parameter(0)
  ROOT %sort = f32[10,10,1000] sort(%input), dimensions={2}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
