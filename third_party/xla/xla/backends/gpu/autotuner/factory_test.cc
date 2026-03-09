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

#include "xla/backends/gpu/autotuner/factory.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using autotuner::Backend;

struct FactoryTestParams {
  std::vector<Backend> names;
  int expected_num_backends;
  bool run_on_cuda = true;
  bool run_on_rocm = true;
};

class FactoryTest : public xla::HloHardwareIndependentTestBase,
                    public ::testing::WithParamInterface<FactoryTestParams> {
 protected:
  se::Platform* platform_;
  std::unique_ptr<Compiler> compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;

  FactoryTest()
      : platform_(se::PlatformManager::PlatformWithName(
                      absl::AsciiStrToUpper(
                          PlatformUtil::CanonicalPlatformName("gpu").value()))
                      .value()),
        compiler_(xla::Compiler::GetForPlatform(platform_->id()).value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_) {}
};

TEST_P(FactoryTest, GetCodegenBackends) {
  const auto& device = stream_executor_->GetDeviceDescription();
  bool is_cuda = device.gpu_compute_capability().IsCuda();
  bool is_rocm = device.gpu_compute_capability().IsRocm();
  if ((GetParam().run_on_cuda && is_cuda) ||
      (GetParam().run_on_rocm && is_rocm)) {
    auto& registry =
        stream_executor::PlatformObjectRegistry::GetGlobalRegistry();
    TF_ASSERT_OK_AND_ASSIGN(
        const GetCodegenBackends::Type& get_codegen_backends,
        registry.FindObject<GetCodegenBackends>(platform_->id()));
    mlir::MLIRContext mlir_context;
    AliasInfo alias_info;
    xla::RegisterSymbolicExprStorage(&mlir_context);
    std::vector<std::unique_ptr<CodegenBackend>> backends =
        get_codegen_backends(stream_executor_, &debug_options_, compiler_.get(),
                             &target_config_, &alias_info, &mlir_context,
                             GetParam().names);
    EXPECT_EQ(backends.size(), GetParam().expected_num_backends);
  } else {
    GTEST_SKIP() << "Skipping test for platform " << platform_->id();
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, FactoryTest,
    ::testing::Values(
        FactoryTestParams{{}, 7, /*run_on_cuda=*/true, /*run_on_cuda=*/false},
        FactoryTestParams{{}, 4, /*run_on_cuda=*/false, /*run_on_cuda=*/true},
        FactoryTestParams{{Backend::TRITON}, 1},
        FactoryTestParams{{Backend::TRITON, Backend::CUBLAS},
                          2,
                          /*run_on_cuda=*/true,
                          /*run_on_cuda=*/false},
        FactoryTestParams{{Backend::TRITON, Backend::ROCBLAS},
                          2,
                          /*run_on_cuda=*/false,
                          /*run_on_cuda=*/true}));

}  // namespace
}  // namespace gpu
}  // namespace xla
