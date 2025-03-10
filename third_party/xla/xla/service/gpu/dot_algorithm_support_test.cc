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

#include <string>
#include <tuple>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::stream_executor::SemanticVersion;
using ::testing::Combine;
using ::testing::HasSubstr;
using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;

enum class BackendRestriction {
  kNoRestriction = 0,
  kTritonOnly = 1,
};

std::string BackendRestrictionToString(BackendRestriction backend_restriction) {
  switch (backend_restriction) {
    case BackendRestriction::kNoRestriction:
      return "no_restriction";
    case BackendRestriction::kTritonOnly:
      return "triton_only";
  }
}

struct Sizes {
  int contracting_size;
  int non_contracting_size;
};

struct TestParams {
  using TupleType = std::tuple<PrecisionConfig::Algorithm, PrimitiveType,
                               PrimitiveType, se::CudaComputeCapability,
                               SemanticVersion, BackendRestriction, Sizes>;

  PrecisionConfig::Algorithm algorithm;
  PrimitiveType input_storage_type;
  PrimitiveType output_storage_type;
  se::CudaComputeCapability min_cuda_capability;
  SemanticVersion min_rocm_version;
  BackendRestriction backend_restriction;
  Sizes sizes;

  explicit TestParams(TupleType t)
      : algorithm(std::get<0>(t)),
        input_storage_type(std::get<1>(t)),
        output_storage_type(std::get<2>(t)),
        min_cuda_capability(std::get<3>(t)),
        min_rocm_version(std::get<4>(t)),
        backend_restriction(std::get<5>(t)),
        sizes(std::get<6>(t)) {}
};

std::string TestParamsToString(
    const TestParamInfo<TestParams::TupleType> &info) {
  const TestParams params(info.param);
  return absl::StrFormat(
      "%s_with_input_%s_output_%s_from_cc_%d_%d_rocm_%d%d_%s_c_%d_nc_%d",
      AlgorithmToString(params.algorithm),
      primitive_util::LowercasePrimitiveTypeName(params.input_storage_type),
      primitive_util::LowercasePrimitiveTypeName(params.output_storage_type),
      params.min_cuda_capability.major, params.min_cuda_capability.minor,
      params.min_rocm_version.major(), params.min_rocm_version.minor(),
      BackendRestrictionToString(params.backend_restriction),
      params.sizes.contracting_size, params.sizes.non_contracting_size);
}

// These are integration tests.
// TODO(tdanyluk): Consider checking somehow directly if the correct algorithms
// are called / emitted. Currently the emitters should decline unsupported
// algorithms, but maybe we could check this directly.
//
// We pass the tuple type instead of the struct to WithParamInterface, to avoid
// the usage of ::testing::ConvertGenerator, which broke the build in some OSS
// configurations.
class DotAlgorithmSupportTest
    : public HloTestBase,
      public WithParamInterface<TestParams::TupleType> {
 public:
  se::DeviceDescription GetDeviceDescription() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
  se::GpuComputeCapability GetGpuComputeCapability() {
    return GetDeviceDescription().gpu_compute_capability();
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    // Setting this explicitly to make sure that we also test the case when the
    // dot's dimensions are under the rewrite size threshold:
    // (2 * non_contracting_size * contracting_size < threshold).
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(100);
    return debug_options;
  }
};

// A parametrized test that checks if an algorithm is supported, with the given
// input and output storage types, from a given cuda capability.
TEST_P(DotAlgorithmSupportTest, AlgorithmIsSupportedFromCudaCapability) {
  const TestParams params(GetParam());

  const std::string hlo_text = absl::Substitute(
      R"(
    HloModule test

    ENTRY test {
      x = $1[$4,$3] parameter(0)
      y = $1[$3,$4] parameter(1)

      ROOT out = $2[$4,$4] dot(x, y),
                 lhs_contracting_dims={1},
                 rhs_contracting_dims={0},
                 algorithm=$0
    }
  )",
      AlgorithmToString(params.algorithm),
      primitive_util::LowercasePrimitiveTypeName(params.input_storage_type),
      primitive_util::LowercasePrimitiveTypeName(params.output_storage_type),
      params.sizes.contracting_size, params.sizes.non_contracting_size);

  bool is_algorithm_supported = false;
  auto gpu_cc = GetGpuComputeCapability();

  if (const auto *ccc = std::get_if<se::CudaComputeCapability>(&gpu_cc)) {
    is_algorithm_supported = ccc->IsAtLeast(params.min_cuda_capability.major,
                                            params.min_cuda_capability.minor);
  } else if (const auto *rcc =
                 std::get_if<se::RocmComputeCapability>(&gpu_cc)) {
    is_algorithm_supported = rcc->gfx9_mi100_or_later();
    if (GetDeviceDescription().runtime_version() < params.min_rocm_version &&
        (params.input_storage_type == F8E5M2 ||
         params.input_storage_type == F8E4M3FN) &&
        params.output_storage_type == BF16) {
      GTEST_SKIP() << "TODO: Unsupported F8 to BF16 in ROCm version < 6.3";
    }
    if (params.backend_restriction == BackendRestriction::kTritonOnly) {
      GTEST_SKIP() << "TODO: Triton unsupported in ROCm";
    }
  }
  if (is_algorithm_supported) {
    EXPECT_TRUE(Run(hlo_text));

    if (params.backend_restriction == BackendRestriction::kTritonOnly) {
      MatchOptimizedHlo(hlo_text, R"(
        ;CHECK: ENTRY
        ;CHECK: ROOT
        ;CHECK-SAME: kCustom
        ;CHECK-SAME: "triton_gemm_config"
    )");
    }
  } else {
    EXPECT_THAT(Run(hlo_text).message(), HasSubstr("Unsupported algorithm"));
  }
}

using PC = PrecisionConfig;
using CC = se::CudaComputeCapability;

INSTANTIATE_TEST_SUITE_P(
    F8E5M2Tests, DotAlgorithmSupportTest,
    Combine(Values(PC::ALG_DOT_ANY_F8_ANY_F8_F32,
                   PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM),
            Values(F8E5M2), Values(F8E5M2, F16, BF16, F32), Values(CC(8, 9)),
            Values(SemanticVersion{6, 3, 0}),
            Values(BackendRestriction::kNoRestriction),
            Values(Sizes{32, 32}, Sizes{16, 2})),
    TestParamsToString);

INSTANTIATE_TEST_SUITE_P(
    F8E4M3FNTests, DotAlgorithmSupportTest,
    Combine(Values(PC::ALG_DOT_ANY_F8_ANY_F8_F32,
                   PC::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM),
            Values(F8E4M3FN), Values(F8E4M3FN, F16, BF16, F32),
            Values(CC(8, 9)), Values(SemanticVersion{6, 3, 0}),
            Values(BackendRestriction::kNoRestriction),
            Values(Sizes{32, 32}, Sizes{16, 2})),
    TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotF16F16F32Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_F16_F16_F32), Values(F16),
                                 Values(F16, F32), Values(CC(0, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotF32ForBf16Bf16F32Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32), Values(F32),
                                 Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotBf16Bf16F32X3Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32_X3),
                                 Values(F32), Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotBf16Bf16F32X6Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32_X6),
                                 Values(F32), Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotBf16Bf16F32X9Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_BF16_BF16_F32_X9),
                                 Values(F32), Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotTf32Tf32F32Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_TF32_TF32_F32), Values(F32),
                                 Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotTf32Tf32F32X3Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_TF32_TF32_F32_X3),
                                 Values(F32), Values(F32), Values(CC(8, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotF32F32F32Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_F32_F32_F32), Values(F32),
                                 Values(F32), Values(CC(0, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

INSTANTIATE_TEST_SUITE_P(DotF64F64F64Tests, DotAlgorithmSupportTest,
                         Combine(Values(PC::ALG_DOT_F64_F64_F64), Values(F64),
                                 Values(F64), Values(CC(0, 0)),
                                 Values(SemanticVersion{6, 0, 0}),
                                 Values(BackendRestriction::kNoRestriction),
                                 Values(Sizes{32, 32}, Sizes{16, 2})),
                         TestParamsToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
