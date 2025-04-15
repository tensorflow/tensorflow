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

#include <variant>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/service/overload.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class FloatSupportTest : public HloTestBase {
 public:
  const se::GpuComputeCapability& GetGpuComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }
};

class FloatSupportTestWithCublas : public FloatSupportTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = FloatSupportTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }
};

class FloatSupportTestWithTriton : public FloatSupportTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = FloatSupportTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_F(FloatSupportTestWithCublas, MixedTypeDotIsNotUpcasted) {
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = bf16[32,32] parameter(0)
  p1 = bf16[32,32] parameter(1)
  ROOT d = f32[32,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK-NOT: convert
; CHECK: __cublas
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{1e-6, 1e-6}));
}

TEST_F(FloatSupportTestWithTriton, MixedTypeDotWithBF16IsNotUpcasted) {
  bool skip_test = std::visit(
      Overload{[](const se::CudaComputeCapability& cc) {
                 return !cc.IsAtLeast(se::CudaComputeCapability::kAmpere);
               },
               [](const se::RocmComputeCapability&) { return true; }},
      GetGpuComputeCapability());

  if (skip_test) {
    GTEST_SKIP() << "Not supported on this GPU architecture";
  }

  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = bf16[32,32] parameter(0)
  p1 = bf16[32,32] parameter(1)
  ROOT d = f32[32,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK-NOT: convert
; CHECK: __triton
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{1e-6, 1e-6}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
