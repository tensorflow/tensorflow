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

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/transforms/estimate_cub_scratch_size.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

std::string GetNumpyOrderComparator(
    const absl::string_view type_name, const absl::string_view direction,
    const bool argsort = false, const absl::string_view index_type_name = "") {
  std::string params = absl::Substitute(
      "  lhs = $0[] parameter(0)\n  rhs = $0[] parameter(1)", type_name);
  if (argsort) {
    absl::StrAppend(&params,
                    absl::Substitute("\n  p2 = $0[] parameter(2)\n  p3 = $0[] "
                                     "parameter(3)",
                                     index_type_name));
  }

  constexpr char kBody[] = R"(
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = $0[] constant(nan)
  c_zero = $0[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = $0[] select(lhs_is_zero, c_zero, lhs)
  lhs_no_neg_zero_or_nan = $0[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = $0[] select(rhs_is_zero, c_zero, rhs)
  rhs_no_neg_zero_or_nan = $0[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  ROOT compare = pred[] compare(lhs_no_neg_zero_or_nan, rhs_no_neg_zero_or_nan), direction=$1, type=TOTALORDER
)";

  return absl::StrCat("numpy_order_comparator {\n", params,
                      absl::Substitute(kBody, type_name, direction), "}");
}

class SortRewriterTestBase
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  void SetUp() override {
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>::SetUp();
    SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAlways);
  }

  bool RunModuleAndPass(HloModule* module) {
    auto cloned = module->Clone();
    bool changed = SortRewriter(TestGpuDeviceInfo::CudaOrRocmDeviceInfo())
                       .Run(module)
                       .value();
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

class SortRewriterTest
    : public SortRewriterTestBase,
      public ::testing::WithParamInterface<std::tuple<PrimitiveType, bool>> {};

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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Parameter()),
          0)));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Parameter()),
          0)));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Parameter()),
          0)));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

// Sort a pair of S32 tensors, keys go first.
TEST_F(SortRewriterTest, SortS32Pairs) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_key = s32[] parameter(0)
  %rhs_key = s32[] parameter(1)
  %lhs_value = s32[] parameter(2)
  %rhs_value = s32[] parameter(3)
  ROOT %lt = pred[] compare(%lhs_key, %rhs_key), direction=LT
}

ENTRY %main {
  %input_keys = s32[1000] parameter(0)
  %input_values = s32[1000] parameter(1)
  ROOT %sort = (s32[1000], s32[1000]) sort(%input_keys, %input_values),
      dimensions={0}, is_stable=true, to_apply=%compare
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, NoRewriteDynamicSize) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = u8[] parameter(0)
  %rhs = u8[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = u8[100,<=100] parameter(0)
  ROOT %sort = u8[100,<=100] sort(%input), dimensions={1}, to_apply=%compare
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, NoRewriteDynamicBatch) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = u8[] parameter(0)
  %rhs = u8[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = u8[<=100,100] parameter(0)
  ROOT %sort = u8[<=100,100] sort(%input), dimensions={1}, to_apply=%compare
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Small shapes do not see improvement from CUB sort.
TEST_F(SortRewriterTest, NoRewriteSmallSize) {
  SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAuto);
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, H100Heuristic) {
  SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAuto);
  constexpr char kHloTmpl[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[$0,100000] parameter(0)
  ROOT %sort = f32[$0,100000] sort(%input), dimensions={1}, to_apply=%compare
})";

  if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm)) {
    GTEST_SKIP() << "Skipping CUDA-specific test";
  }
  auto pass = SortRewriter(TestGpuDeviceInfo::RTXH100SXMDeviceInfo());

  // Batch 1
  std::string hlo = absl::Substitute(kHloTmpl, "1");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Batch 3
  hlo = absl::Substitute(kHloTmpl, "3");
  ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);

  // Batch 70
  hlo = absl::Substitute(kHloTmpl, "70");
  ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(SortRewriterTest, A100Heuristic) {
  SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAuto);
  constexpr char kHloTmpl[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[$0,100000] parameter(0)
  ROOT %sort = f32[$0,100000] sort(%input), dimensions={1}, to_apply=%compare
})";

  if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm)) {
    GTEST_SKIP() << "Skipping CUDA-specific test";
  }
  auto pass = SortRewriter(TestGpuDeviceInfo::RTXA6000DeviceInfo());

  // Batch 1
  std::string hlo = absl::Substitute(kHloTmpl, "1");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Batch 3
  hlo = absl::Substitute(kHloTmpl, "3");
  ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);

  // Batch 31
  hlo = absl::Substitute(kHloTmpl, "31");
  ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Parameter()),
          0)));
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Parameter()),
          0)));
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
    // CHECK: %[[CC:.*]] = (u16[1000]{0}, u8[{{[0-9]+}}]{0}) custom-call({{.*}}), custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize", metadata={op_type="sort" op_name="sort" source_file="path/to/test.cc" source_line=68}, backend_config={"descending":true}
  )";

  bool is_cuda = test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuCuda);
  auto device_list = [is_cuda]() -> std::vector<se::DeviceDescription> {
    if (is_cuda) {
      return {TestGpuDeviceInfo::RTXA6000DeviceInfo(),
              TestGpuDeviceInfo::RTXH100SXMDeviceInfo()};
    } else {
      return {TestGpuDeviceInfo::AMDMI210DeviceInfo(),
              TestGpuDeviceInfo::AMDRX7900DeviceInfo()};
    }
  };

  for (const auto& device_desc : device_list()) {
    RunAndFilecheckHloRewrite(kHlo, SortRewriter(device_desc),
                              kExpectedPattern);
  }
}

TEST_P(SortRewriterTest, SortNumpyOrder) {
  auto [dtype, direction] = GetParam();
  std::string type_name = primitive_util::LowercasePrimitiveTypeName(dtype);
  std::string direction_str = direction ? "LT" : "GT";

  std::string hlo_str =
      absl::StrCat(GetNumpyOrderComparator(type_name, direction_str),
                   absl::Substitute(R"(
ENTRY main {
  p = $0[16,128] parameter(0)
  ROOT sort = $0[16,128] sort(p), dimensions={1}, is_stable=true, to_apply=numpy_order_comparator
})",
                                    type_name));

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunModuleAndPass(module.get())) << module->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                        m::Op(), m::Parameter()),
          1)))
      << module->ToString();
}

INSTANTIATE_TEST_SUITE_P(
    SortRewriterTest, SortRewriterTest,
    ::testing::Combine(::testing::Values(F16, BF16, F32, F64),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<SortRewriterTest::ParamType>& info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          std::get<1>(info.param) ? "_asc" : "_desc");
    });

TEST_F(SortRewriterTest, NoRewriteLargeInputNumpyOrder) {
  std::string hlo_str = absl::StrCat(
      GetNumpyOrderComparator("f32", "LT", /*argsort=*/true, "s32"),
      R"(
ENTRY main {
  // 300,000,000 elements * 8 bytes (4 for f32 + 4 for s32) = 2.4 GB > 2 GB
  p = f32[300000000] parameter(0)
  i = s32[300000000] parameter(1)
  ROOT sort = (f32[300000000], s32[300000000]) sort(p, i), dimensions={0}, is_stable=true, to_apply=numpy_order_comparator
})");

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  // Should not rewrite because input size is too large.
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

TEST_F(SortRewriterTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortMode(), SortRewriter::Mode::kAlways);
}

class SortRewriterArgsortTest
    : public SortRewriterTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, bool, PrimitiveType>> {};

TEST_P(SortRewriterArgsortTest, SortNumpyOrderArgsort) {
  auto [key_type, ascending, index_type] = GetParam();
  std::string type_name = primitive_util::LowercasePrimitiveTypeName(key_type);
  std::string direction_str = ascending ? "LT" : "GT";
  std::string index_type_name =
      primitive_util::LowercasePrimitiveTypeName(index_type);

  std::string hlo_str =
      absl::StrCat(GetNumpyOrderComparator(type_name, direction_str,
                                           /*argsort=*/true, index_type_name),
                   absl::Substitute(R"(
ENTRY main {
  p = $0[16,128] parameter(0)
  i = $1[16,128] iota(), iota_dimension=1
  ROOT sort = ($0[16,128], $1[16,128]) sort(p, i), dimensions={1}, is_stable=true, to_apply=numpy_order_comparator
})",
                                    type_name, index_type_name));

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));
  bool changed = RunModuleAndPass(module.get());
  bool should_use_cub = key_type != F64;
  if (should_use_cub) {
    EXPECT_TRUE(changed) << module->ToString();
    EXPECT_THAT(module->entry_computation()->instructions(),
                ::testing::Contains(GmockMatch(m::CustomCall(
                    {kCubDeviceRadixSortUnassignedScratchSizeTarget}))));
  } else {
    EXPECT_FALSE(changed) << module->ToString();
  }
}

INSTANTIATE_TEST_SUITE_P(
    SortRewriterArgsort, SortRewriterArgsortTest,
    ::testing::Combine(::testing::Values(F16, BF16, F32, F64),
                       ::testing::Bool(), ::testing::Values(S16, S32)),
    [](const ::testing::TestParamInfo<SortRewriterArgsortTest::ParamType>&
           info) {
      PrimitiveType key_type = std::get<0>(info.param);
      bool ascending = std::get<1>(info.param);
      PrimitiveType index_type = std::get<2>(info.param);
      bool should_use_cub = key_type != F64;
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(key_type),
          ascending ? "_asc" : "_desc", "_",
          primitive_util::LowercasePrimitiveTypeName(index_type),
          should_use_cub ? "_cub" : "_nocub");
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
