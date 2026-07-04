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

#include "xla/backends/autotuner/persistent_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

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

using ::tsl::protobuf::util::MessageDifferencer;

class MockPersistentCache : public PersistentCache {
 public:
  MockPersistentCache(AutotuneCacheContext cache_ctx, CacheMode mode,
                      KeyMatchingMode matching_mode,
                      std::vector<autotuner::AutotuneEntry>* entries)
      : PersistentCache(std::move(cache_ctx), mode, matching_mode),
        entries_(entries) {}

 protected:
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) override {
    std::vector<autotuner::AutotuneEntry> result;
    for (const auto& entry : *entries_) {
      if (MessageDifferencer::Equivalent(entry.key().target(), target_key)) {
        result.push_back(entry);
      }
    }
    return result;
  }

  absl::Status Write(const autotuner::AutotuneEntry& entry) override {
    for (auto& existing : *entries_) {
      if (MessageDifferencer::Equivalent(existing.key(), entry.key())) {
        *existing.mutable_value() = entry.value();
        return absl::OkStatus();
      }
    }
    entries_->push_back(entry);
    return absl::OkStatus();
  }

 private:
  std::vector<autotuner::AutotuneEntry>* entries_;
};

const char* kHlo1 = R"(
HloModule module1
ENTRY entry {
  p0 = f32[10,10]{1,0} parameter(0)
  p1 = f32[10,10]{1,0} parameter(1)
  ROOT dot = f32[10,10]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

class PersistentCacheTest : public HloHardwareIndependentTestBase {
 protected:
  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }

  std::vector<autotuner::AutotuneEntry> shared_entries_;
};

TEST_F(PersistentCacheTest, BasicInsertAndLookup) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  MockPersistentCache cache(CreateCacheContext(), CacheMode::kReadWrite,
                            KeyMatchingMode::kStrict, &shared_entries_);
  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(dot, config));

  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  EXPECT_EQ(cache.GetCacheStats().hits, 1);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);
}

TEST_F(PersistentCacheTest, StrictModeRejection) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext();
  MockPersistentCache cache(cache_ctx, CacheMode::kReadWrite,
                            KeyMatchingMode::kStrict, &shared_entries_);
  AutotunerCacheInterface::Config config = CreateTestConfig();
  EXPECT_OK(cache.Insert(dot, config));

  // Create a new context with different backend version to change
  // codegen_version.
  AutotuneCacheContext cache_ctx2 =
      CreateCacheContext("v1.0", {{autotuner::Backend::TRITON, "triton_v2"}});
  MockPersistentCache cache2(cache_ctx2, CacheMode::kReadOnly,
                             KeyMatchingMode::kStrict, &shared_entries_);

  std::optional<AutotunerCacheInterface::Config> result = cache2.Lookup(dot);
  EXPECT_FALSE(result.has_value());
}

TEST_F(PersistentCacheTest, LooseModeMatching) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  AutotuneCacheContext cache_ctx = CreateCacheContext();
  MockPersistentCache cache(cache_ctx, CacheMode::kReadWrite,
                            KeyMatchingMode::kLoose, &shared_entries_);
  AutotunerCacheInterface::Config config =
      CreateTestConfig(autotuner::Backend::TRITON);
  EXPECT_OK(cache.Insert(dot, config));

  // Change codegen version, but keep backend version same.
  // Add another backend to change codegen_version, but keep TRITON version
  // same.
  AutotuneCacheContext cache_ctx2 =
      CreateCacheContext("v1.0", {{autotuner::Backend::TRITON, "triton_v1"},
                                  {autotuner::Backend::CUDNN, "cudnn_v1"}});
  MockPersistentCache cache2(cache_ctx2, CacheMode::kReadOnly,
                             KeyMatchingMode::kLoose, &shared_entries_);

  std::optional<AutotunerCacheInterface::Config> result = cache2.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);
}

TEST_F(PersistentCacheTest, CacheReadOnlyConstraints) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  MockPersistentCache cache_ro(CreateCacheContext(), CacheMode::kReadOnly,
                               KeyMatchingMode::kStrict, &shared_entries_);
  AutotunerCacheInterface::Config config = CreateTestConfig();
  EXPECT_EQ(cache_ro.Insert(dot, config).code(),
            absl::StatusCode::kPermissionDenied);
}

}  // namespace
}  // namespace xla
