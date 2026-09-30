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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
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

// Returns the canonical GPU platform.
se::Platform* GetGpuPlatform() {
  absl::StatusOr<std::string> name = PlatformUtil::CanonicalPlatformName("gpu");
  CHECK_OK(name.status());
  absl::StatusOr<se::Platform*> platform =
      se::PlatformManager::PlatformWithName(absl::AsciiStrToUpper(*name));
  CHECK_OK(platform.status());
  return *platform;
}

// Returns the compiler registered for `platform`.
std::unique_ptr<Compiler> GetCompilerForPlatform(se::Platform* platform) {
  absl::StatusOr<std::unique_ptr<Compiler>> compiler =
      Compiler::GetForPlatform(platform->id());
  CHECK_OK(compiler.status());
  return *std::move(compiler);
}

// Returns the first device executor of `platform`.
se::StreamExecutor* GetStreamExecutor(se::Platform* platform) {
  absl::StatusOr<se::StreamExecutor*> stream_executor =
      platform->ExecutorForDevice(0);
  CHECK_OK(stream_executor.status());
  return *stream_executor;
}

class FactoryTest : public xla::HloHardwareIndependentTestBase,
                    public ::testing::WithParamInterface<FactoryTestParams> {
 protected:
  se::Platform* platform_;
  std::unique_ptr<Compiler> compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  stream_executor::StreamExecutorAddressAllocator allocator_;

  FactoryTest()
      : platform_(GetGpuPlatform()),
        compiler_(GetCompilerForPlatform(platform_)),
        stream_executor_(GetStreamExecutor(platform_)),
        target_config_(stream_executor_),
        allocator_(stream_executor_) {}
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
        get_codegen_backends(
            stream_executor_, &allocator_, &debug_options_, compiler_.get(),
            &target_config_, &alias_info, &mlir_context,
            /*shape_size_fn=*/[](const Shape&) { return 0; }, GetParam().names,
            /*thread_pool=*/nullptr, /*mlir_context_pool=*/nullptr);
    EXPECT_EQ(backends.size(), GetParam().expected_num_backends);
  } else {
    GTEST_SKIP() << "Skipping test for platform " << platform_->id();
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, FactoryTest,
    ::testing::Values(
        FactoryTestParams{{}, 6, /*run_on_cuda=*/true, /*run_on_rocm=*/false},
        FactoryTestParams{{}, 6, /*run_on_cuda=*/false, /*run_on_rocm=*/true},
        FactoryTestParams{{Backend::TRITON}, 1},
        FactoryTestParams{{Backend::TRITON, Backend::CUBLASLT},
                          2,
                          /*run_on_cuda=*/true,
                          /*run_on_rocm=*/false},
        FactoryTestParams{{Backend::TRITON, Backend::HIPBLASLT},
                          2,
                          /*run_on_cuda=*/false,
                          /*run_on_rocm=*/true}));

}  // namespace
}  // namespace gpu
}  // namespace xla
