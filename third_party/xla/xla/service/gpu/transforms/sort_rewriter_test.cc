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

#include "xla/service/gpu/transforms/sort_rewriter.h"

#include <utility>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
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

class SortRewriterTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    SortRewriter::SetSortSizeThresholdForTestingOnly(
        0);  // Always use CUB sort.
  }

  bool RunModuleAndPass(HloModule* module) {
    auto cloned = module->Clone();
    bool changed = SortRewriter().Run(module).value();
    if (changed) {
      // Here we run an end to end test to make sure that SortRewriter does
      // not introduce an incorrect rewrite. To do this, we need to clone the
      // original module because the interpreter cannot process the already
      // optimized module.
      EXPECT_TRUE(RunAndCompare(std::move(cloned), ErrorSpec{0, 0}));
    }
    return changed;
  }

  void ExpectDirection(const HloInstruction* instruction, bool descending) {
    auto config = instruction->backend_config<xla::SortOptions>();
    EXPECT_EQ(config->descending(), descending);
  }
};

// Basic sort: ascending.
TEST_F(SortRewriterTest, SortKeysLessThan) {
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
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: descending.
TEST_F(SortRewriterTest, SortKeysGreaterThan) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/true);
}

// Comparer swaps the parameter order -> direction is reversed.
TEST_F(SortRewriterTest, SortKeysGreaterThanSwapped) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(1)
  %rhs = f32[] parameter(0)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Sort a pair of tensors, keys go first.
TEST_F(SortRewriterTest, SortPairs) {
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
  %input_keys = u32[1000] parameter(0)
  %input_values = f32[1000] parameter(1)
  ROOT %sort = (u32[1000], f32[1000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

// Sort a pair of tensors, keys go last.
TEST_F(SortRewriterTest, SortPairsSwapped) {
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
  %input_values = f32[1000] parameter(0)
  %input_keys = u32[1000] parameter(1)
  ROOT %sort = (f32[1000], u32[1000]) sort(%input_values, %input_keys),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 1),
                                  m::GetTupleElement(m::CustomCall(), 0))));
}

// CUB sort doesn't support more than two tensors.
TEST_F(SortRewriterTest, NoRewriteManyTensors) {
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
  %input1 = f32[1000] parameter(0)
  %input2 = f64[1000] parameter(1)
  %input3 = u64[1000] parameter(2)
  ROOT %sort = (f32[1000], f64[1000], u64[1000]) sort(%input1, %input2, %input3),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Only 1D shapes are supported.
TEST_F(SortRewriterTest, NoRewriteNonMinorSortDimension) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000,4] parameter(0)
  ROOT %sort = f32[1000,4] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Kernels are compiled for a subset of types.
TEST_F(SortRewriterTest, NoRewriteUnsupportedType) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = pred[] parameter(0)
  %rhs = pred[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = pred[1000] parameter(0)
  ROOT %sort = pred[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Comparer must be a simple function.
TEST_F(SortRewriterTest, NoRewriteComplexComparer) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %lhs_scaled = f32[] multiply(%lhs, f32[] constant(2))
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs_scaled, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Comparer must use adjacent input values.
TEST_F(SortRewriterTest, NoRewriteMixedKeysValues) {
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
  %input_keys = u32[1000] parameter(0)
  %input_values = u32[1000] parameter(1)
  ROOT %sort = (u32[1000], u32[1000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Small shapes do not see improvement from CUB sort.
TEST_F(SortRewriterTest, NoRewriteSmallSize) {
  SortRewriter::SetSortSizeThresholdForTestingOnly(16385);
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100] parameter(0)
  ROOT %sort = f32[100] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Basic sort: with batch dimension.
TEST_F(SortRewriterTest, SortWithBatchDim) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[10,100] parameter(0)
  ROOT %sort = f32[10,100] sort(%input), dimensions={1}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: with multiple batch dimensions.
TEST_F(SortRewriterTest, SortWithMultipleBatchDims) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[10,10,10] parameter(0)
  ROOT %sort = f32[10,10,10] sort(%input), dimensions={2}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

TEST_F(SortRewriterTest, SortPreservesMetadata) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = u16[] parameter(0)
  %rhs = u16[] parameter(1)
  ROOT %cmp_lr = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %inputs = u16[1000] parameter(0)
  ROOT %sort = u16[1000] sort(%inputs),
      dimensions={0}, to_apply=%compare, metadata={op_type="sort" op_name="sort" source_file="path/to/test.cc" source_line=68}
})";
  constexpr char kExpectedPattern[] = R"(
    // CHECK: %[[CC:.*]] = (u16[1000]{0}, u8[1]{0}) custom-call({{.*}}), custom_call_target="__cub$DeviceRadixSort", metadata={op_type="sort" op_name="sort" source_file="path/to/test.cc" source_line=68}, backend_config={"descending":true}
  )";
  RunAndFilecheckHloRewrite(kHlo, SortRewriter(), kExpectedPattern);
}

TEST_F(SortRewriterTest, SortNumpyOrderLessThan) {
  constexpr char kHlo[] = R"(
numpy_order_comparator {
  lhs = bf16[] parameter(0)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = bf16[] constant(nan)
  c_zero = bf16[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = bf16[] select(lhs_is_zero, c_zero, lhs)
  lhs_no_neg_zero_or_nan = bf16[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  rhs = bf16[] parameter(1)
  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = bf16[] select(rhs_is_zero, c_zero, rhs)
  rhs_no_neg_zero_or_nan = bf16[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  ROOT compare.20017 = pred[] compare(lhs_no_neg_zero_or_nan, rhs_no_neg_zero_or_nan), direction=LT, type=TOTALORDER
}

ENTRY main {
  p = bf16[128] parameter(0)
  ROOT sort = bf16[128] sort(p), dimensions={0}, is_stable=true, to_apply=numpy_order_comparator
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, SortNumpyOrderGreaterThan) {
  constexpr char kHlo[] = R"(
numpy_order_comparator {
  lhs = bf16[] parameter(0)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = bf16[] constant(nan)
  c_zero = bf16[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = bf16[] select(lhs_is_zero, c_zero, lhs)
  lhs_no_neg_zero_or_nan = bf16[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  rhs = bf16[] parameter(1)
  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = bf16[] select(rhs_is_zero, c_zero, rhs)
  rhs_no_neg_zero_or_nan = bf16[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  ROOT compare.20017 = pred[] compare(lhs_no_neg_zero_or_nan, rhs_no_neg_zero_or_nan), direction=GT, type=TOTALORDER
}

ENTRY main {
  p = bf16[128] parameter(0)
  ROOT sort = bf16[128] sort(p), dimensions={0}, is_stable=true, to_apply=numpy_order_comparator
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortSizeThreshold(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
