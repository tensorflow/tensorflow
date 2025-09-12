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

#include "xla/backends/gpu/autotuner/legacy_cache.h"

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
#include "xla/tsl/protobuf/dnn.pb.h"

namespace xla {
namespace gpu {
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

class LegacyCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&test_dir_));
    TF_ASSERT_OK(tsl::Env::Default()->CreateDir(test_dir_));
  }

  void TearDown() override {
    int64_t undeleted_files, undeleted_dirs;
    tsl::Env::Default()
        ->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string test_dir_;
  DebugOptions::AutotuneCacheMode mode_ =
      DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE;
  std::string device_name_ = "test_device";
  se::DeviceDescription device_desc_ =
      CreateDummyDeviceDescription(device_name_);

  std::unique_ptr<HloInstruction> CreateDummyInstr(const std::string& name) {
    return HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(std::hash<std::string>()(name)));
  }

  Config CreateDummyTritonConfig() {
    Config config;
    config.codegen_backend_name = "Triton";
    config.backend_config.PackFrom(AutotuneResult::TritonGemmKey());
    return config;
  }

  Config CreateDummyCublasConfig() {
    Config config;
    config.codegen_backend_name = "Cublas";
    config.backend_config.PackFrom(AutotuneResult::GemmKey());
    return config;
  }

  Config CreateDummyCudnnConfig() {
    Config config;
    config.codegen_backend_name = "Cudnn";
    config.backend_config.PackFrom(stream_executor::dnn::AlgorithmProto());
    return config;
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

TEST_F(LegacyCacheTest, CreateEmpty) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo");
  EXPECT_THAT(cache.Lookup(instr.get()), Eq(std::nullopt));
}

TEST_F(LegacyCacheTest, InsertAndLookupTriton) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo1");
  Config config = CreateDummyTritonConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupCublas) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo2");
  Config config = CreateDummyCublasConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupCudnn) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo3");
  Config config = CreateDummyCudnnConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupForUnsupportedBackend) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo4");
  Config config;
  config.codegen_backend_name = "UnsupportedBackend";

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Eq(std::nullopt));
}

TEST_F(LegacyCacheTest, PersistAcrossInstances) {
  auto instr = CreateDummyInstr("hlo5");
  Config config = CreateDummyTritonConfig();

  // Create cache, insert, and let it save.
  {
    auto cache = LegacyCache(test_dir_, mode_, device_desc_);
    TF_ASSERT_OK(cache.Insert(instr.get(), config));
  }

  // Create a new cache, which should load from disk.
  {
    auto cache = LegacyCache(test_dir_, mode_, device_desc_);
    EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
  }
}

TEST_F(LegacyCacheTest, LoadWithDifferentDevice) {
  auto instr = CreateDummyInstr("hlo6");
  Config config = CreateDummyTritonConfig();

  // Create cache, insert, and let it save.
  {
    auto cache = LegacyCache(test_dir_, mode_,
                             CreateDummyDeviceDescription("test_device"));
    TF_ASSERT_OK(cache.Insert(instr.get(), config));
  }

  // Create a new cache with different device, should not load the entry.
  {
    auto cache = LegacyCache(test_dir_, mode_,
                             CreateDummyDeviceDescription("other_device"));
    EXPECT_THAT(cache.Lookup(instr.get()), Eq(std::nullopt));
  }
}

TEST_F(LegacyCacheTest, OnlyInsertOncePerHlo) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo7");

  Config config = CreateDummyTritonConfig();
  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));

  Config another_config = CreateDummyCublasConfig();
  TF_ASSERT_OK(cache.Insert(instr.get(), another_config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
