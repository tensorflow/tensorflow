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

#include "xla/backends/autotuner/file_based_autotuner_cache.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "google/protobuf/any.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using Config = ::xla::AutotunerCacheInterface::Config;
using ::testing::Eq;
using ::testing::Optional;

// Helper to create a dummy DeviceDescription.
se::DeviceDescription CreateDummyDeviceDescription(
    std::string name = "test_device") {
  se::DeviceDescription desc;
  desc.set_name(name);
  desc.set_device_vendor("NVIDIA");
  desc.set_platform_version("CUDA 12.0");
  desc.set_gpu_compute_capability(se::CudaComputeCapability(8, 0));
  desc.set_core_count(108);
  desc.set_clock_rate_ghz(name == "test_device" ? 1.98 : 2.00);
  desc.set_memory_bandwidth(1000e9);
  desc.set_l2_cache_size(50 * 1024 * 1024);
  return desc;
}

class FileBasedAutotunerCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    test_dir_ = tsl::io::JoinPath(testing::TempDir(), test_info->name());
    // Clean up any previous test runs.
    int64_t undeleted_files, undeleted_dirs;
    tsl::Env::Default()
        ->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
    ASSERT_OK(tsl::Env::Default()->CreateDir(test_dir_));
  }

  void TearDown() override {
    int64_t undeleted_files, undeleted_dirs;
    tsl::Env::Default()
        ->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string test_dir_;

  FileBasedCacheConfig GetConfig(
      se::DeviceDescription device_desc,
      FileBasedCacheConfig::CacheMode mode =
          FileBasedCacheConfig::CacheMode::READ_WRITE) {
    FileBasedCacheConfig config;
    config.autotune_cache_dir = test_dir_;
    config.autotune_cache_mode = mode;
    config.device_desc = device_desc;
    return config;
  }

  std::unique_ptr<HloInstruction> CreateDummyInstr(const std::string& name) {
    return HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(std::hash<std::string>()(name)));
  }

  google::protobuf::Any CreateDummyBackendConfig() {
    google::protobuf::Any any;
    return any;
  }
};

// Matcher for Config.
MATCHER_P(ConfigEq, expected_config, "") {
  const Config& actual_config = arg;
  if (actual_config.codegen_backend_name !=
      expected_config.codegen_backend_name) {
    *result_listener << "codegen_backend mismatch: expected "
                     << expected_config.codegen_backend_name << ", got "
                     << actual_config.codegen_backend_name;
    return false;
  }
  // Compare backend_config (google::protobuf::Any)
  if (actual_config.backend_config.type_url() !=
      expected_config.backend_config.type_url()) {
    *result_listener << "backend_config type_url mismatch: expected "
                     << expected_config.backend_config.type_url() << ", got "
                     << actual_config.backend_config.type_url();
    return false;
  }
  if (actual_config.backend_config.value() !=
      expected_config.backend_config.value()) {
    *result_listener << "backend_config value mismatch";
    return false;
  }
  return true;
}

TEST_F(FileBasedAutotunerCacheTest, CreateEmpty) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, FileBasedAutotunerCache::Create(
                      GetConfig(CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
  auto instr = CreateDummyInstr("hlo1");
  EXPECT_THAT(cache->Lookup(instr.get()), Eq(std::nullopt));
}

TEST_F(FileBasedAutotunerCacheTest, InsertAndLookup) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, FileBasedAutotunerCache::Create(
                      GetConfig(CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
  auto instr = CreateDummyInstr("hlo1");
  Config config;
  config.codegen_backend_name = "TestBackend";
  config.backend_config = CreateDummyBackendConfig();

  ASSERT_OK(cache->Insert(instr.get(), config));
  EXPECT_THAT(cache->Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(FileBasedAutotunerCacheTest, SaveAndLoad) {
  auto instr = CreateDummyInstr("hlo2");
  Config config;
  config.codegen_backend_name = "TestBackend";
  config.backend_config = CreateDummyBackendConfig();

  // Create cache, insert, and let it save.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    ASSERT_OK(cache->Insert(instr.get(), config));
  }

  // Create a new cache, which should load from disk.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    EXPECT_THAT(cache->Lookup(instr.get()), Optional(ConfigEq(config)));
  }
}

TEST_F(FileBasedAutotunerCacheTest, LoadWithDifferentDevice) {
  auto instr = CreateDummyInstr("hlo2");
  Config config;
  config.codegen_backend_name = "TestBackend";
  config.backend_config = CreateDummyBackendConfig();

  // Create cache, insert, and let it save.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    ASSERT_OK(cache->Insert(instr.get(), config));
  }

  // Create a new cache with different device, should not load the entry.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription("other_device"),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    EXPECT_THAT(cache->Lookup(instr.get()), Eq(std::nullopt));
  }
}

TEST_F(FileBasedAutotunerCacheTest, LoadWithDifferentVersion) {
  auto instr = CreateDummyInstr("hlo2");
  Config config;
  config.codegen_backend_name = "TestBackend";
  config.backend_config = CreateDummyBackendConfig();

  // Create cache, insert, and let it save.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    ASSERT_OK(cache->Insert(instr.get(), config));
  }

  // Create a new cache with different version, should not load the entry.
  {
    auto cache_config = GetConfig(CreateDummyDeviceDescription(),
                                  FileBasedCacheConfig::CacheMode::READ_WRITE);
    cache_config.cache_version = "2";
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(cache_config));
    EXPECT_THAT(cache->Lookup(instr.get()), Eq(std::nullopt));
  }
}

TEST_F(FileBasedAutotunerCacheTest, ReadOnlyMode) {
  auto instr = CreateDummyInstr("hlo3");
  Config config;
  config.codegen_backend_name = "TestBackend";
  config.backend_config = CreateDummyBackendConfig();

  // Create in READ_WRITE mode to pre-populate the cache file.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    ASSERT_OK(cache->Insert(instr.get(), config));
  }

  // Create in READ mode.
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, FileBasedAutotunerCache::Create(
                      GetConfig(CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ)));
  // Lookup should work.
  EXPECT_THAT(cache->Lookup(instr.get()), Optional(ConfigEq(config)));

  // Insert a new entry.
  auto instr2 = CreateDummyInstr("hlo4");
  Config config2;
  config2.codegen_backend_name = "AnotherBackend";
  config2.backend_config = CreateDummyBackendConfig();
  ASSERT_OK(cache->Insert(instr2.get(), config2));
  EXPECT_THAT(cache->Lookup(instr2.get()), Eq(std::nullopt));

  // Create a new cache, key2 should not be present as it wasn't saved.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto cache2,
                            FileBasedAutotunerCache::Create(GetConfig(
                                CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
    EXPECT_THAT(cache2->Lookup(instr.get()), Optional(ConfigEq(config)));
    EXPECT_THAT(cache2->Lookup(instr2.get()), Eq(std::nullopt));
  }
}

TEST_F(FileBasedAutotunerCacheTest, OverwriteEntry) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto cache, FileBasedAutotunerCache::Create(
                      GetConfig(CreateDummyDeviceDescription(),
                                FileBasedCacheConfig::CacheMode::READ_WRITE)));
  auto instr = CreateDummyInstr("hlo5");

  Config config1;
  config1.codegen_backend_name = "BackendV1";
  config1.backend_config = CreateDummyBackendConfig();
  ASSERT_OK(cache->Insert(instr.get(), config1));
  EXPECT_THAT(cache->Lookup(instr.get()), Optional(ConfigEq(config1)));

  Config config2;
  config2.codegen_backend_name = "BackendV2";
  config2.backend_config = CreateDummyBackendConfig();
  ASSERT_OK(cache->Insert(instr.get(), config2));
  EXPECT_THAT(cache->Lookup(instr.get()), Optional(ConfigEq(config2)));
}

}  // namespace
}  // namespace xla
