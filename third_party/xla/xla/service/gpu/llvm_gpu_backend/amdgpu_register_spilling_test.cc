/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

static std::string RemoveLLExtension(const std::string& filename) {
  return filename.substr(0, filename.find(".ll"));
}

// Test parameter structure
struct SpillingTestParam {
  std::string ir_filename;         // IR file to compile
  bool fail_on_spilling;           // Flag value
  absl::StatusCode expected_code;  // Expected status code
  std::string expected_substring;  // Expected substring in error (if any)
};

class AMDGPURegisterSpillingTest
    : public ::testing::TestWithParam<SpillingTestParam> {
 protected:
  // Helper to load IR module from test data
  std::unique_ptr<llvm::Module> LoadTestModule(llvm::LLVMContext* context,
                                               const std::string& filename) {
    return LoadIRModule(
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                          "llvm_gpu_backend", "tests_data", filename),
        context);
  }

  // Helper to compile with given debug options
  absl::StatusOr<std::vector<uint8_t>> CompileModule(
      llvm::Module* module, const std::string& module_id,
      bool fail_on_spilling) {
    DebugOptions debug_options;
    debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(
        fail_on_spilling);

    module->setModuleIdentifier(module_id);

    return amdgpu::CompileToHsaco(
        module, se::GpuComputeCapability{se::RocmComputeCapability{"gfx1100"}},
        debug_options, module_id);
  }
};

TEST_P(AMDGPURegisterSpillingTest, CompileTest) {
  const SpillingTestParam& param = GetParam();
  llvm::LLVMContext context;

  auto module = LoadTestModule(&context, param.ir_filename);
  ASSERT_NE(module, nullptr);

  // Generate module ID from filename and flag state
  std::string module_id =
      RemoveLLExtension(param.ir_filename) +
      (param.fail_on_spilling ? "_fail_on_spilling" : "_allow_spilling");

  auto result = CompileModule(module.get(), module_id, param.fail_on_spilling);

  EXPECT_EQ(result.status().code(), param.expected_code)
      << "IR: " << param.ir_filename
      << ", Flag: " << (param.fail_on_spilling ? "enabled" : "disabled")
      << ", Status: " << result.status().message();

  if (!param.expected_substring.empty()) {
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr(param.expected_substring))
        << "IR: " << param.ir_filename;
  }
}

INSTANTIATE_TEST_SUITE_P(
    RegisterSpillingTests, AMDGPURegisterSpillingTest,
    ::testing::Values(
        SpillingTestParam{"amdgpu_no_spills.ll",
                          /*fail_on_spilling=*/true, absl::StatusCode::kOk, ""},
        SpillingTestParam{"amdgpu_vgpr_spills.ll",
                          /*fail_on_spilling=*/false, absl::StatusCode::kOk,
                          ""},
        SpillingTestParam{"amdgpu_vgpr_spills.ll",
                          /*fail_on_spilling=*/true,
                          absl::StatusCode::kCancelled, "register spilling"},
        SpillingTestParam{"amdgpu_sgpr_spills.ll",
                          /*fail_on_spilling=*/false, absl::StatusCode::kOk,
                          ""},
        SpillingTestParam{"amdgpu_sgpr_spills.ll",
                          /*fail_on_spilling=*/true, absl::StatusCode::kOk, ""},
        SpillingTestParam{"amdgpu_dynamic_stack.ll",
                          /*fail_on_spilling=*/true,
                          absl::StatusCode::kCancelled, "stack usage"}),
    [](const ::testing::TestParamInfo<SpillingTestParam>& info) {
      return RemoveLLExtension(info.param.ir_filename) +
             (info.param.fail_on_spilling ? "_fail_on_spilling"
                                          : "_allow_spilling");
    });

}  // namespace
}  // namespace xla::gpu
