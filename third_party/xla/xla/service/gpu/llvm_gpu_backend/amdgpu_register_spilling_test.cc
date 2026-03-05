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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_stats.h"
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
  // Module stats expectations (only checked when expected_code == kOk).
  bool expect_stats_empty = true;
  std::string expected_kernel_name;  // Kernel name to look up in module_stats
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
  absl::StatusOr<amdgpu::HsacoResult> CompileModule(
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

  // When compilation succeeds, verify module_stats.
  if (result.ok()) {
    const ModuleStats& stats = result->module_stats;
    if (param.expect_stats_empty) {
      EXPECT_TRUE(stats.empty()) << "IR: " << param.ir_filename
                                 << " — expected empty module_stats but got "
                                 << stats.size() << " entries";
    } else {
      EXPECT_FALSE(stats.empty()) << "IR: " << param.ir_filename
                                  << " — expected non-empty module_stats";
      auto it = stats.find(param.expected_kernel_name);
      ASSERT_NE(it, stats.end())
          << "IR: " << param.ir_filename << " — expected kernel '"
          << param.expected_kernel_name << "' in module_stats";
      EXPECT_GT(it->second.store_bytes_spilled, 0);
      EXPECT_GT(it->second.load_bytes_spilled, 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    RegisterSpillingTests, AMDGPURegisterSpillingTest,
    ::testing::Values(
        SpillingTestParam{"amdgpu_no_spills.ll",
                          /*fail_on_spilling=*/true, absl::StatusCode::kOk, "",
                          /*expect_stats_empty=*/true},
        SpillingTestParam{"amdgpu_vgpr_spills.ll",
                          /*fail_on_spilling=*/false, absl::StatusCode::kOk, "",
                          /*expect_stats_empty=*/false,
                          /*expected_kernel_name=*/"high_register_pressure"},
        SpillingTestParam{"amdgpu_vgpr_spills.ll",
                          /*fail_on_spilling=*/true,
                          absl::StatusCode::kCancelled, "register spilling"},
        SpillingTestParam{"amdgpu_sgpr_spills.ll",
                          /*fail_on_spilling=*/false, absl::StatusCode::kOk, "",
                          /*expect_stats_empty=*/true},
        SpillingTestParam{"amdgpu_sgpr_spills.ll",
                          /*fail_on_spilling=*/true, absl::StatusCode::kOk, "",
                          /*expect_stats_empty=*/true},
        // Dynamic stack (uses_dynamic_stack=true) with private_segment_size=0
        // produces empty module_stats since there are no actual spill bytes.
        // The Tier 1 hard-fail path (fail_on_spilling=true) handles this case
        // independently via HasStackUsage().
        SpillingTestParam{"amdgpu_dynamic_stack.ll",
                          /*fail_on_spilling=*/false, absl::StatusCode::kOk, "",
                          /*expect_stats_empty=*/true},
        SpillingTestParam{"amdgpu_dynamic_stack.ll",
                          /*fail_on_spilling=*/true,
                          absl::StatusCode::kCancelled, "stack usage"}),
    [](const ::testing::TestParamInfo<SpillingTestParam>& info) {
      return RemoveLLExtension(info.param.ir_filename) +
             (info.param.fail_on_spilling ? "_fail_on_spilling"
                                          : "_allow_spilling");
    });

// Verify that the HSACO cache-hit path returns correct and identical
// module_stats.
TEST(AMDGPUCacheHitModuleStatsTest, CacheHitReturnsIdenticalModuleStats) {
  llvm::LLVMContext context;
  const std::string module_id = "cache_hit_stats_test";
  auto load_module = [&context]() {
    return LoadIRModule(
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                          "llvm_gpu_backend", "tests_data",
                          "amdgpu_vgpr_spills.ll"),
        &context);
  };
  auto compile = [&module_id](llvm::Module* module) {
    DebugOptions debug_options;
    module->setModuleIdentifier(module_id);
    return amdgpu::CompileToHsaco(
        module, se::GpuComputeCapability{se::RocmComputeCapability{"gfx1100"}},
        debug_options, module_id);
  };

  // First compilation has cache miss.
  auto module1 = load_module();
  ASSERT_NE(module1, nullptr);
  auto result1 = compile(module1.get());
  ASSERT_TRUE(result1.ok()) << result1.status().message();

  // Second compilation has cache hit.
  auto module2 = load_module();
  ASSERT_NE(module2, nullptr);
  auto result2 = compile(module2.get());
  ASSERT_TRUE(result2.ok()) << result2.status().message();

  // Both results should have the same module_stats.
  const ModuleStats& stats1 = result1->module_stats;
  const ModuleStats& stats2 = result2->module_stats;

  EXPECT_EQ(stats1.size(), stats2.size())
      << "Cache hit should return same number of kernel stats entries";

  for (const auto& [name, ks1] : stats1) {
    auto it = stats2.find(name);
    ASSERT_NE(it, stats2.end())
        << "Cache hit missing kernel '" << name << "' in module_stats";
    EXPECT_EQ(ks1.store_bytes_spilled, it->second.store_bytes_spilled)
        << "Mismatch in store_bytes_spilled for kernel '" << name << "'";
    EXPECT_EQ(ks1.load_bytes_spilled, it->second.load_bytes_spilled)
        << "Mismatch in load_bytes_spilled for kernel '" << name << "'";
  }
}

}  // namespace
}  // namespace xla::gpu
