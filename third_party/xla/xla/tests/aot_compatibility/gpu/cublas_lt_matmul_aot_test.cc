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

// AOT compatibility of CublasLtMatmulThunk (GemmRewriter + cuBLASLt), on 1
// GPU. The cuBLASLt algorithm is pinned per arch, so the golden does not depend
// on the cuBLASLt heuristic.

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "cublas_lt_matmul_aot_test";

class CublasLtMatmulAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  CublasLtMatmulAotTest() : AotCompatibilityTest(GetParam()) {}
};

// The heuristic's first pick depends on the CUDA version the test is built with
// (on B200, algo_id 16 at 13.2 and 23 at the default 12.8), so a golden left to
// the heuristic would too. These descriptors are heuristic picks from H100
// and B200; numerical_impl_flags must match what cuBLASLt reports for the
// algorithm, or the runtime cannot rebuild it.
absl::string_view PinnedCublasLtAlgorithm(absl::string_view target_name) {
  if (absl::EndsWith(target_name, "_h100")) {
    return "algo_id: 63 tile_id: 11 split_k: 1 stages_id: 15 "
           "numerical_impl_flags: 262657";
  }
  if (absl::EndsWith(target_name, "_b200") ||
      absl::EndsWith(target_name, "_gb200")) {
    return "algo_id: 16 split_k: 1 numerical_impl_flags: 524801";
  }
  // A new arch needs its own descriptor, picked from a heuristic run on it.
  LOG(FATAL) << "No pinned cuBLASLt algorithm for " << target_name;
}

// Forces the CUBLASLT backend with the pinned algorithm. The workspace size is
// the one the heuristic configs request.
std::string ForcedCublasLtConfig(absl::string_view target_name) {
  return absl::StrCat(
      "backend: CUBLASLT backend_config { gemm { algorithm: 0 "
      "autotune_workspace_size: 33554432 algorithm_proto { cublaslt { ",
      PinnedCublasLtAlgorithm(target_name), " } } } }");
}

// Triton GEMM off and cuBLASLt on, so GemmRewriter lowers the dot to a
// CublasLtMatmulThunk.
TEST_P(CublasLtMatmulAotTest, CublasLtMatmul) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      lhs = f32[2,2] parameter(0)
      rhs = f32[2,2] parameter(1)
      ROOT dot = f32[2,2] dot(lhs, rhs), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  debug_options.set_xla_gpu_enable_cublaslt(true);
  debug_options.set_xla_force_config(
      ForcedCublasLtConfig(GetParam().target_name));
  // Fail instead of falling back to the heuristic if the runtime cannot
  // rebuild the serialized algorithm.
  debug_options.set_xla_gpu_require_exact_blas_lt_algorithm(true);
  // Lets the 2x2 dot reach GemmRewriter instead of the elemental emitter.
  debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
  Literal lhs = LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}});
  Literal rhs = LiteralUtil::CreateR2<float>({{5, 6}, {7, 8}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&lhs, &rhs}));
  LiteralTestUtil::ExpectR2Near<float>({{19, 22}, {43, 50}}, result,
                                       ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, CublasLtMatmulAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, CublasLtMatmulAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
