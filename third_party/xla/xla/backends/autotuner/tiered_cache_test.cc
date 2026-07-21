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

#include "xla/backends/autotuner/tiered_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/directory_cache.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/backends/autotuner/local_cache.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"

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
    std::string explicit_version = "v1.0",
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

class TieredCacheTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    ASSERT_OK_AND_ASSIGN(
        auto temp,
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
    temp_dir_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(std::move(temp));
    cache_dir_ = temp_dir_->path();
  }

  void TearDown() override {
    temp_dir_.reset();
    HloHardwareIndependentTestBase::TearDown();
  }

  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }

  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_dir_;
  std::string cache_dir_;
  LocalCacheStorage local_storage_;
};

TEST_F(TieredCacheTest, PropagationOnLookupMiss) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  AutotunerCacheInterface::Config config = CreateTestConfig();

  // 1. Populate the persistent DirectoryCache directly.
  {
    DirectoryCache temp_dir_cache(cache_ctx, cache_dir_, CacheMode::kReadWrite,
                                  KeyMatchingMode::kStrict);
    EXPECT_OK(temp_dir_cache.Insert(instr1, config));
  }

  // 2. Create the TieredCache delegates.
  auto persistent_cache = std::make_unique<DirectoryCache>(
      cache_ctx, cache_dir_, CacheMode::kReadWrite, KeyMatchingMode::kStrict);
  auto local_cache = std::make_unique<LocalCache>(
      cache_ctx, persistent_cache->GetKeyMatchingMode(), &local_storage_);
  TieredCache cache(std::move(local_cache), std::move(persistent_cache));

  // Verify that the statistics start empty.
  EXPECT_EQ(cache.GetCacheStats().hits, 0);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  // 3. Lookup. The local in-memory cache should miss, but the persistent
  // cache should hit and propagate to the in-memory cache.
  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  // Persistent hit should contribute to hits, and misses should remain 0
  // despite the local cache miss.
  EXPECT_EQ(cache.GetCacheStats().hits, 1);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  // 4. Query again. Should now hit the local in-memory cache directly.
  std::optional<AutotunerCacheInterface::Config> result2 = cache.Lookup(instr1);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(result2->codegen_backend, autotuner::Backend::TRITON);

  // Stats should now be 2 hits (1 local, 1 persistent), and 0 misses.
  EXPECT_EQ(cache.GetCacheStats().hits, 2);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);
}

TEST_F(TieredCacheTest, TotalMissStat) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  auto persistent_cache = std::make_unique<DirectoryCache>(
      cache_ctx, cache_dir_, CacheMode::kReadWrite, KeyMatchingMode::kStrict);
  auto local_cache = std::make_unique<LocalCache>(
      cache_ctx, persistent_cache->GetKeyMatchingMode(), &local_storage_);
  TieredCache cache(std::move(local_cache), std::move(persistent_cache));

  // Verify stats start at 0.
  EXPECT_EQ(cache.GetCacheStats().hits, 0);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  // Lookup. Should miss in both local and persistent.
  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  EXPECT_FALSE(result.has_value());

  // Hits should remain 0, and misses should be 1.
  EXPECT_EQ(cache.GetCacheStats().hits, 0);
  EXPECT_EQ(cache.GetCacheStats().misses, 1);
}

TEST_F(TieredCacheTest, DualInsertion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  auto persistent_cache = std::make_unique<DirectoryCache>(
      cache_ctx, cache_dir_, CacheMode::kReadWrite, KeyMatchingMode::kStrict);
  auto local_cache = std::make_unique<LocalCache>(
      cache_ctx, persistent_cache->GetKeyMatchingMode(), &local_storage_);
  TieredCache cache(std::move(local_cache), std::move(persistent_cache));
  AutotunerCacheInterface::Config config = CreateTestConfig();

  // Insert config using TieredCache.
  EXPECT_OK(cache.Insert(instr1, config));

  // The local delegate and persistent delegate both should contain the mapping.
  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  // Verify that another cache instance pointing to same file path also hits.
  DirectoryCache persistent_check(cache_ctx, cache_dir_, CacheMode::kReadOnly,
                                  KeyMatchingMode::kStrict);
  std::optional<AutotunerCacheInterface::Config> result2 =
      persistent_check.Lookup(instr1);
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(result2->codegen_backend, autotuner::Backend::TRITON);
}

}  // namespace
}  // namespace xla
