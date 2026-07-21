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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/directory_store.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/backends/autotuner/in_memory_store.h"
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
    std::string explicit_version = "v1.0", int core_count = 108,
    std::vector<std::pair<autotuner::Backend, std::string>> backends_info =
        {{autotuner::Backend::TRITON, "triton_v1"}},
    std::string device_name = "test_gpu") {
  stream_executor::DeviceDescription device_description;
  device_description.set_name(device_name);
  device_description.set_core_count(core_count);
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

// A store that records how it was called, used to assert that TieredCache
// builds a single target key and shares it across all tiers.
class RecordingStore : public AutotuneCacheStore {
 public:
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) override {
    read_count++;
    last_target_key = target_key;
    return std::vector<autotuner::AutotuneEntry>{};
  }
  absl::Status Write(const autotuner::AutotuneEntry& entry) override {
    write_count++;
    return absl::OkStatus();
  }
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> ReadAll() override {
    return std::vector<autotuner::AutotuneEntry>{};
  }
  CacheMode GetMode() const override { return CacheMode::kReadWrite; }

  int read_count = 0;
  int write_count = 0;
  autotuner::AutotuneTargetKey last_target_key;
};

class TieredCacheTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    InMemoryStore::Clear();
    ASSERT_OK_AND_ASSIGN(
        auto temp,
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
    temp_dir_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(std::move(temp));
    cache_dir_ = temp_dir_->path();
  }

  void TearDown() override { temp_dir_.reset(); }

  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }

  std::unique_ptr<TieredCache> MakeTieredCache(const AutotuneCacheContext& ctx,
                                               bool with_directory) {
    auto primary = std::make_unique<InMemoryStore>();
    std::unique_ptr<AutotuneCacheStore> secondary = nullptr;
    if (with_directory) {
      secondary =
          std::make_unique<DirectoryStore>(cache_dir_, CacheMode::kReadWrite);
    }
    return std::make_unique<TieredCache>(
        ctx, KeyMatchingMode::kLoose, std::move(primary), std::move(secondary));
  }

  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_dir_;
  std::string cache_dir_;
};

TEST_F(TieredCacheTest, PromotesToHotterTierOnColdHit) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  // 1. Populate only the persistent directory tier.
  {
    auto primary =
        std::make_unique<DirectoryStore>(cache_dir_, CacheMode::kReadWrite);
    TieredCache dir_only(cache_ctx, KeyMatchingMode::kLoose,
                         std::move(primary));
    EXPECT_OK(dir_only.Insert(instr1, config));
  }

  // 2. Fresh in-memory tier + the persistent directory tier.
  std::unique_ptr<TieredCache> cache =
      MakeTieredCache(cache_ctx, /*with_directory=*/true);

  EXPECT_EQ(cache->GetCacheStats().hits, 0);
  EXPECT_EQ(cache->GetCacheStats().misses, 0);

  // 3. First lookup misses in-memory, hits directory, and promotes.
  std::optional<AutotunerCacheInterface::Config> result = cache->Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);
  EXPECT_EQ(cache->GetCacheStats().hits, 1);
  EXPECT_EQ(cache->GetCacheStats().misses, 0);

  // The entry should have been promoted into the in-memory tier.
  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       InMemoryStore().ReadAll());
  EXPECT_THAT(entries, testing::SizeIs(1));

  // 4. Second lookup hits the hot in-memory tier.
  EXPECT_TRUE(cache->Lookup(instr1).has_value());
  EXPECT_EQ(cache->GetCacheStats().hits, 2);
  EXPECT_EQ(cache->GetCacheStats().misses, 0);
}

TEST_F(TieredCacheTest, TotalMissIncrementsMissStat) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  std::unique_ptr<TieredCache> cache =
      MakeTieredCache(cache_ctx, /*with_directory=*/true);

  EXPECT_FALSE(cache->Lookup(instr1).has_value());
  EXPECT_EQ(cache->GetCacheStats().hits, 0);
  EXPECT_EQ(cache->GetCacheStats().misses, 1);
}

TEST_F(TieredCacheTest, InsertWritesToAllTiers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  std::unique_ptr<TieredCache> cache =
      MakeTieredCache(cache_ctx, /*with_directory=*/true);
  EXPECT_OK(cache->Insert(instr1, config));

  // In-memory tier populated.
  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       InMemoryStore().ReadAll());
  EXPECT_THAT(entries, testing::SizeIs(1));

  // Persistent tier populated: a separate directory-only cache can find it.
  auto primary =
      std::make_unique<DirectoryStore>(cache_dir_, CacheMode::kReadOnly);
  TieredCache dir_only(cache_ctx, KeyMatchingMode::kLoose, std::move(primary));
  EXPECT_TRUE(dir_only.Lookup(instr1).has_value());
}

TEST_F(TieredCacheTest, ProcessWideInMemorySharing) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  InMemoryStore::Clear();

  // Cache 1 uses the global storage (default constructor) plus directory.
  auto primary1 = std::make_unique<InMemoryStore>();
  auto secondary1 =
      std::make_unique<DirectoryStore>(cache_dir_, CacheMode::kReadWrite);
  TieredCache cache1(cache_ctx, KeyMatchingMode::kLoose, std::move(primary1),
                     std::move(secondary1));
  EXPECT_OK(cache1.Insert(instr1, config));

  // Cache 2 uses the same global storage; lookup should hit the hot tier.
  auto primary2 = std::make_unique<InMemoryStore>();
  auto secondary2 =
      std::make_unique<DirectoryStore>(cache_dir_, CacheMode::kReadWrite);
  TieredCache cache2(cache_ctx, KeyMatchingMode::kLoose, std::move(primary2),
                     std::move(secondary2));

  EXPECT_TRUE(cache2.Lookup(instr1).has_value());
  EXPECT_EQ(cache2.GetCacheStats().hits, 1);
}

TEST_F(TieredCacheTest, SerializeDeserializeRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  std::unique_ptr<TieredCache> cache =
      MakeTieredCache(cache_ctx, /*with_directory=*/false);
  EXPECT_OK(cache->Insert(instr1, config));

  ASSERT_OK_AND_ASSIGN(std::string serialized, cache->Serialize({instr1}));

  // Clear the global store so we can prove deserialization repopulates it.
  InMemoryStore::Clear();

  std::unique_ptr<TieredCache> cache2 =
      MakeTieredCache(cache_ctx, /*with_directory=*/false);
  EXPECT_OK(cache2->Deserialize(serialized));
  EXPECT_TRUE(cache2->Lookup(instr1).has_value());
}

TEST_F(TieredCacheTest, HashesOnceAndSharesTargetKeyAcrossTiers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");

  auto store0 = std::make_unique<RecordingStore>();
  auto store1 = std::make_unique<RecordingStore>();
  RecordingStore* store0_ptr = store0.get();
  RecordingStore* store1_ptr = store1.get();

  TieredCache cache(cache_ctx, KeyMatchingMode::kLoose, std::move(store0),
                    std::move(store1));

  // A single lookup that misses everywhere should read each tier exactly once
  // with an identical target key (proving the fingerprint was computed once).
  EXPECT_FALSE(cache.Lookup(instr1).has_value());
  EXPECT_EQ(store0_ptr->read_count, 1);
  EXPECT_EQ(store1_ptr->read_count, 1);
  EXPECT_EQ(store0_ptr->last_target_key.device(),
            store1_ptr->last_target_key.device());
  EXPECT_EQ(store0_ptr->last_target_key.explicit_version(),
            store1_ptr->last_target_key.explicit_version());
  EXPECT_EQ(store0_ptr->last_target_key.hlo_fingerprint(),
            store1_ptr->last_target_key.hlo_fingerprint());
  EXPECT_FALSE(store0_ptr->last_target_key.hlo_fingerprint().empty());
}

TEST_F(TieredCacheTest, StrictMatchingRequiresSameOptions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  auto primary = std::make_unique<InMemoryStore>();
  TieredCache cache(cache_ctx, KeyMatchingMode::kStrict, std::move(primary));

  EXPECT_OK(cache.Insert(instr1, config));
  EXPECT_TRUE(cache.Lookup(instr1).has_value());

  // Modify debug options to change the options fingerprint.
  DebugOptions debug_options = module->config().debug_options();
  debug_options.set_xla_gpu_enable_fast_min_max(
      !debug_options.xla_gpu_enable_fast_min_max());
  module->mutable_config().set_debug_options(debug_options);

  EXPECT_FALSE(cache.Lookup(instr1).has_value());
}

TEST_F(TieredCacheTest, LooseMatchingIgnoresOptions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotuneCacheContext cache_ctx = CreateCacheContext("v1.0");
  AutotunerCacheInterface::Config config = CreateTestConfig();

  auto primary = std::make_unique<InMemoryStore>();
  TieredCache cache(cache_ctx, KeyMatchingMode::kLoose, std::move(primary));

  EXPECT_OK(cache.Insert(instr1, config));
  EXPECT_TRUE(cache.Lookup(instr1).has_value());

  // Modify debug options.
  DebugOptions debug_options = module->config().debug_options();
  debug_options.set_xla_gpu_enable_fast_min_max(
      !debug_options.xla_gpu_enable_fast_min_max());
  module->mutable_config().set_debug_options(debug_options);

  // Still hits.
  EXPECT_TRUE(cache.Lookup(instr1).has_value());
}

TEST_F(TieredCacheTest, LooseMatchingBackendVersionFallback) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();
  AutotunerCacheInterface::Config config =
      CreateTestConfig(autotuner::Backend::TRITON);

  InMemoryStore::Clear();

  // 1. Insert with Context A (Triton "triton_v1").
  AutotuneCacheContext ctx_A = CreateCacheContext(
      "v1.0", 108, {{autotuner::Backend::TRITON, "triton_v1"}}, "test_gpu");
  {
    auto primary = std::make_unique<InMemoryStore>();
    TieredCache cache_A(ctx_A, KeyMatchingMode::kLoose, std::move(primary));
    EXPECT_OK(cache_A.Insert(instr1, config));
  }

  // 2. Lookup with Context B (Triton "triton_v1", cuBLAS "cublas_v2").
  // Codegen version differs, but Triton version matches. Should HIT.
  AutotuneCacheContext ctx_B =
      CreateCacheContext("v1.0", 108,
                         {{autotuner::Backend::TRITON, "triton_v1"},
                          {autotuner::Backend::CUBLASLT, "cublas_v2"}},
                         "test_gpu");
  {
    auto primary = std::make_unique<InMemoryStore>();
    TieredCache cache_B(ctx_B, KeyMatchingMode::kLoose, std::move(primary));
    EXPECT_TRUE(cache_B.Lookup(instr1).has_value());
  }

  // 3. Lookup with Context C (Triton "triton_v2", cuBLAS "cublas_v2").
  // Triton version differs. Should MISS.
  AutotuneCacheContext ctx_C =
      CreateCacheContext("v1.0", 108,
                         {{autotuner::Backend::TRITON, "triton_v2"},
                          {autotuner::Backend::CUBLASLT, "cublas_v2"}},
                         "test_gpu");
  {
    auto primary = std::make_unique<InMemoryStore>();
    TieredCache cache_C(ctx_C, KeyMatchingMode::kLoose, std::move(primary));
    EXPECT_FALSE(cache_C.Lookup(instr1).has_value());
  }
}

}  // namespace
}  // namespace xla
