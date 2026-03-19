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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/tests/hlo_legacy_gpu_test_base.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class GemmRewriteAllocationTest : public HloLegacyGpuTestBase {
 public:
  void CheckNumberOfAllocations(const std::string& hlo,
                                int expected_number_of_allocations) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo));
    if (allocator_ == nullptr) {
      allocator_ =
          std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
              backend().default_stream_executor());
    }
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler()->RunBackend(std::move(optimized_module),
                               backend().default_stream_executor(),
                               allocator_.get()));
    GpuExecutable* gpu_executable =
        static_cast<GpuExecutable*>(executable.get());
    ASSERT_EQ(gpu_executable->GetAllocations().size(),
              expected_number_of_allocations);
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloLegacyGpuTestBase::GetDebugOptionsForTest();
    // Make sure the rewriter does not skip the rewrite for being too small.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }

 private:
  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
};

TEST_F(GemmRewriteAllocationTest, SharedBufferAssignment) {
  const char* hlo_text = R"(
HloModule SharedBufferAssignment

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] add(x, y)
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot, bias)
}

)";

  // Bias should be fused into the multiplication.
  CheckNumberOfAllocations(hlo_text, 4);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
