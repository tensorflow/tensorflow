/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/autotuner/directory_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

const char* kHlo1 = R"(
HloModule module1
ENTRY entry {
  p0 = f32[10,10]{1,0} parameter(0)
  p1 = f32[10,10]{1,0} parameter(1)
  ROOT dot = f32[10,10]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

AutotuneCacheContext CreateCacheContext(
    std::string explicit_version = "v1",
    std::vector<std::pair<autotuner::Backend, std::string>> backends_info = {
        {autotuner::Backend::TRITON, "triton_v1"}}) {
  stream_executor::DeviceDescription device_description;
  device_description.set_name("test_gpu");
  device_description.set_core_count(108);
  device_description.set_clock_rate_ghz(1.41);
  device_description.set_memory_bandwidth(1555000000000);
  device_description.set_l2_cache_size(41943040);

  stream_executor::CudaComputeCapability cuda_cc(8, 0);
  stream_executor::GpuComputeCapability gpu_cc(cuda_cc);
  device_description.set_gpu_compute_capability(gpu_cc);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  for (const auto& [backend, version] : backends_info) {
    backends.push_back(std::make_unique<FakeCodegenBackend>(backend, version));
  }

  return AutotuneCacheContext::Create(device_description, backends,
                                      std::move(explicit_version));
}

class DirectoryCacheTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    ASSERT_OK_AND_ASSIGN(
        auto temp_dir,
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
    temp_dir_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(std::move(temp_dir));
    cache_dir_ = temp_dir_->path();
  }

  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }

  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_dir_;
  std::string cache_dir_;
};

TEST_F(DirectoryCacheTest, BasicInsertAndLookup) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("v1");

  DirectoryCache cache(cache_ctx, cache_dir_, CacheMode::kReadWrite,
                       KeyMatchingMode::kStrict);
  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(instr1, config));

  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  EXPECT_EQ(cache.GetCacheStats().hits, 1);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  // Persistence check: Create a new cache instance pointing to same directory.
  DirectoryCache cache2(cache_ctx, cache_dir_, CacheMode::kReadOnly,
                        KeyMatchingMode::kStrict);
  std::optional<AutotunerCacheInterface::Config> result2 =
      cache2.Lookup(instr1);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(result2->codegen_backend, autotuner::Backend::TRITON);

  // Physical directory structure verification using RE2
  std::vector<std::string> files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(cache_dir_, cache_ctx.device(), "v1", "*.pb"), &files));
  ASSERT_EQ(files.size(), 1);

  // Also check that it serializes AutotuneEntry directly
  std::string serialized;
  EXPECT_OK(tsl::ReadFileToString(tsl::Env::Default(), files[0], &serialized));
  autotuner::AutotuneEntry entry;
  EXPECT_TRUE(entry.ParseFromString(serialized));
  EXPECT_EQ(entry.key().target().device(), cache_ctx.device());
  EXPECT_EQ(entry.key().target().explicit_version(), "v1");
}

TEST_F(DirectoryCacheTest, EmptyExplicitVersionOmitTier) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("");

  DirectoryCache cache(cache_ctx, cache_dir_, CacheMode::kReadWrite,
                       KeyMatchingMode::kStrict);
  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(instr1, config));

  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  // Verification that the version subdirectory tier is omitted.
  std::vector<std::string> files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(cache_dir_, cache_ctx.device(), "*.pb"), &files));
  ASSERT_EQ(files.size(), 1);
}

TEST_F(DirectoryCacheTest, StrictModeRejection) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx1 = CreateCacheContext("v1.0");

  DirectoryCache cache1(cache_ctx1, cache_dir_, CacheMode::kReadWrite,
                        KeyMatchingMode::kStrict);
  AutotunerCacheInterface::Config config = CreateTestConfig();
  EXPECT_OK(cache1.Insert(instr1, config));

  // Create a new context with different backend version to change
  // codegen_version.
  AutotuneCacheContext cache_ctx2 =
      CreateCacheContext("v1.0", {{autotuner::Backend::TRITON, "triton_v2"}});
  DirectoryCache cache2(cache_ctx2, cache_dir_, CacheMode::kReadOnly,
                        KeyMatchingMode::kStrict);

  std::optional<AutotunerCacheInterface::Config> result = cache2.Lookup(instr1);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(cache2.GetCacheStats().misses, 1);
}

TEST_F(DirectoryCacheTest, LooseModeChecks) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx1 = CreateCacheContext("v1.0");

  DirectoryCache cache1(cache_ctx1, cache_dir_, CacheMode::kReadWrite,
                        KeyMatchingMode::kLoose);
  AutotunerCacheInterface::Config config =
      CreateTestConfig(autotuner::Backend::TRITON);
  EXPECT_OK(cache1.Insert(instr1, config));

  // Add another backend to change codegen_version, but keep TRITON version
  // same.
  AutotuneCacheContext cache_ctx2 =
      CreateCacheContext("v1.0", {{autotuner::Backend::TRITON, "triton_v1"},
                                  {autotuner::Backend::CUDNN, "cudnn_v1"}});
  DirectoryCache cache2(cache_ctx2, cache_dir_, CacheMode::kReadOnly,
                        KeyMatchingMode::kLoose);

  std::optional<AutotunerCacheInterface::Config> result = cache2.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  // Lookup with different backend version.
  AutotuneCacheContext cache_ctx3 =
      CreateCacheContext("v1.0", {{autotuner::Backend::TRITON, "triton_v2"}});
  DirectoryCache cache3(cache_ctx3, cache_dir_, CacheMode::kReadOnly,
                        KeyMatchingMode::kLoose);

  std::optional<AutotunerCacheInterface::Config> result3 =
      cache3.Lookup(instr1);
  EXPECT_FALSE(result3.has_value());
}

TEST_F(DirectoryCacheTest, CacheModeConstraints) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  // ReadOnly mode Insert should fail.
  DirectoryCache cache_ro(cache_ctx, cache_dir_, CacheMode::kReadOnly,
                          KeyMatchingMode::kStrict);
  AutotunerCacheInterface::Config config = CreateTestConfig();
  EXPECT_EQ(cache_ro.Insert(instr1, config).code(),
            absl::StatusCode::kPermissionDenied);

  // ReadWrite mode Insert.
  DirectoryCache cache_rw(cache_ctx, cache_dir_, CacheMode::kReadWrite,
                          KeyMatchingMode::kStrict);
  EXPECT_OK(cache_rw.Insert(instr1, config));

  // ReadAppend mode Insert of existing entry should fail.
  DirectoryCache cache_ra(cache_ctx, cache_dir_, CacheMode::kReadAppend,
                          KeyMatchingMode::kStrict);
  EXPECT_EQ(cache_ra.Insert(instr1, config).code(),
            absl::StatusCode::kAlreadyExists);

  // ReadAppend mode Insert of new entry should succeed.
  const HloInstruction* instr2 = instr1->operand(0);
  EXPECT_OK(cache_ra.Insert(instr2, config));
}

}  // namespace
}  // namespace xla
