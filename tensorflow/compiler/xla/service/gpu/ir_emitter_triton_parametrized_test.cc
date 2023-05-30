/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct GemmTestParams {
  PrimitiveType lhs_ty;
  PrimitiveType rhs_ty;
  int m;
  int k;
  int n;
  float aabs = 1e-6;
  float arel = 1e-6;
};

class ParametrizedRewriteTest
    : public GpuCodegenTest,
      public ::testing::WithParamInterface<GemmTestParams> {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_P(ParametrizedRewriteTest, Main) {
  GemmTestParams params = GetParam();
  if ((params.lhs_ty == BF16 || params.rhs_ty == BF16) &&
      !GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  const std::string hlo_string_template = R"(
HloModule m

ENTRY e {
  p0 = $0[$2,$3] parameter(0)
  p0c = $1[$2,$3] convert(p0)
  p1 = $1[$3,$4] parameter(1)
  ROOT _ = $1[$2,$4] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  std::string hlo_string = absl::Substitute(
      hlo_string_template,
      primitive_util::LowercasePrimitiveTypeName(params.lhs_ty),
      primitive_util::LowercasePrimitiveTypeName(params.rhs_ty), params.m,
      params.k, params.n);
  MatchOptimizedHlo(hlo_string, R"(
; CHECK: fusion(%p0, %p1)
; CHECK-SAME: kind=kCustom
; CHECK-SAME: backend_config="{\"block_m\":\"
)");

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{params.aabs, params.arel}));
}

std::string GemmTestParamsParamsToString(
    const ::testing::TestParamInfo<GemmTestParams>& data) {
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data.param.lhs_ty), "_",
      primitive_util::LowercasePrimitiveTypeName(data.param.rhs_ty), "_",
      data.param.m, "_", data.param.k, "_", data.param.n);
}

INSTANTIATE_TEST_SUITE_P(RewriteTestSuite, ParametrizedRewriteTest,
                         ::testing::ValuesIn({
                             GemmTestParams{PRED, F16, 16, 32, 8},
                             GemmTestParams{PRED, BF16, 16, 32, 8},
                             GemmTestParams{PRED, F32, 16, 32, 8, 1e-4, 1e-3},
                             GemmTestParams{S8, F16, 16, 32, 8},
                             GemmTestParams{S8, BF16, 16, 32, 8},
                             GemmTestParams{S8, F32, 16, 32, 8, 5e-2, 1e-2},
                             GemmTestParams{S8, F32, 101, 7, 303, 0.1, 0.1},
                             GemmTestParams{S8, F32, 101, 32, 303, 0.1, 0.1},
                             GemmTestParams{S8, F32, 101, 2048, 303, 0.5, 0.1},
                             GemmTestParams{S8, F32, 101, 2555, 303, 0.5, 0.1},
                             // Is supported but overflows.
                             //  GemmTestParams{S32, F16},
                             GemmTestParams{S16, F16, 30, 19, 12},
                             GemmTestParams{S32, F32, 4, 4, 4, 1, 1e-2},
                             GemmTestParams{F16, BF16, 16, 32, 8},
                             GemmTestParams{F16, F32, 16, 32, 8, 1e-3, 1e-6},
                             GemmTestParams{BF16, F16, 16, 32, 8, 1e-3, 1e-6},
                             GemmTestParams{BF16, F32, 16, 32, 8, 1e-3, 1e-6},
                             // Supported but disabled because narrowing
                             // converts should rather belong to producers.
                             // TODO(b/266862493): Move these to CompareTest.
                             // TritonRewriteTest2Params{S32, BF16},
                             //  TritonRewriteTest2Params{F32, F16},
                             //  TritonRewriteTest2Params{F32, BF16},
                             GemmTestParams{S8, BF16, 24, 40, 8},
                             GemmTestParams{S8, F16, 80, 16, 32},
                             GemmTestParams{F16, F32, 127, 3, 300, 1e-2, 1e-2},
                             GemmTestParams{F16, BF16, 544, 96, 16, 1e-3, 1e-3},
                             GemmTestParams{BF16, F32, 77, 500, 333, 3e-3,
                                            3e-3},
                         }),
                         GemmTestParamsParamsToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
