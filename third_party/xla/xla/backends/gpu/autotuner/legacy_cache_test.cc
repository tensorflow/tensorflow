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
#include <vector>

#include "google/protobuf/any.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using Config = ::xla::AutotunerCacheInterface::Config;
using autotuner::Backend;
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
    config.codegen_backend = Backend::TRITON;
    config.backend_config.PackFrom(AutotuneResult::TritonGemmKey());
    return config;
  }

  Config CreateDummyCublasLtConfig() {
    Config config;
    config.codegen_backend = Backend::CUBLASLT;
    config.backend_config.PackFrom(AutotuneResult::GemmKey());
    return config;
  }

  Config CreateDummyCublasLtFissionConfig() {
    Config config;
    config.codegen_backend = Backend::CUBLASLT_FISSION;
    config.backend_config.PackFrom(AutotuneResult::GemmKey());
    return config;
  }

  Config CreateDummyCudnnConfig() {
    Config config;
    config.codegen_backend = Backend::CUDNN;
    config.backend_config.PackFrom(stream_executor::dnn::AlgorithmProto());
    return config;
  }

  Config CreateDummyCustomKernelFissionConfig() {
    Config config;
    config.codegen_backend = Backend::CUSTOM_KERNEL_FISSION;
    config.backend_config.PackFrom(AutotuneResult::CustomKernelFusionKey());
    return config;
  }

  Config CreateDummyBackendConfig() {
    using DummyOtherConfig = AutotuneResult::CustomKernelFusionKey;
    Config config;
    config.codegen_backend = Backend::CUSTOM_KERNEL_FISSION;
    config.backend_config.PackFrom(DummyOtherConfig());
    return config;
  }
};

// Matcher for Config.
MATCHER_P(ConfigEq, expected_config, "") {
  const Config& actual_config = arg;
  if (actual_config.codegen_backend != expected_config.codegen_backend) {
    *result_listener << "codegen_backend mismatch: expected "
                     << expected_config.codegen_backend << ", got "
                     << actual_config.codegen_backend;
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
  Config config = CreateDummyCublasLtConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupCublasFission) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  constexpr char kHLO[] = R"(
HloModule test_module

fused_computation {
  param.0 = f32[] parameter(0)
  param.1 = f32[] parameter(1)
  ROOT add.0 = f32[] add(param.0, param.1)
}

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT fusion.0 = f32[] fusion(p0, p1), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHLO));
  auto instr = module->entry_computation()->root_instruction();
  Config config = CreateDummyCublasLtFissionConfig();

  TF_ASSERT_OK(cache.Insert(instr, config));
  EXPECT_THAT(cache.Lookup(instr), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupCudnn) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo3");
  Config config = CreateDummyCudnnConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupCustomKernelFission) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo4");
  Config config = CreateDummyCustomKernelFissionConfig();
  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, InsertAndLookupOther) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr = CreateDummyInstr("hlo5");
  Config config = CreateDummyBackendConfig();

  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  std::optional<Config> actual_config = cache.Lookup(instr.get());
  EXPECT_THAT(actual_config, Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, PersistAcrossInstances) {
  auto instr = CreateDummyInstr("hlo6");
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
  auto instr = CreateDummyInstr("hlo7");
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
  auto instr = CreateDummyInstr("hlo8");

  Config config = CreateDummyTritonConfig();
  TF_ASSERT_OK(cache.Insert(instr.get(), config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));

  Config another_config = CreateDummyCublasLtConfig();
  TF_ASSERT_OK(cache.Insert(instr.get(), another_config));
  EXPECT_THAT(cache.Lookup(instr.get()), Optional(ConfigEq(config)));
}

TEST_F(LegacyCacheTest, SerializeAndDeserialize) {
  LegacyCache cache(test_dir_, mode_, device_desc_);
  std::unique_ptr<HloInstruction> instr_1 = CreateDummyInstr("hlo9");
  std::unique_ptr<HloInstruction> instr_2 = CreateDummyInstr("hlo10");
  Config orig_config = CreateDummyTritonConfig();
  TF_ASSERT_OK(cache.Insert(instr_1.get(), orig_config));
  TF_ASSERT_OK(cache.Insert(instr_2.get(), orig_config));

  // Serialize instr_1 to a string.
  std::vector<const HloInstruction*> instructions_to_serialize = {
      instr_1.get()};
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_cache,
                          cache.Serialize(instructions_to_serialize));

  // Overwrite config for both instructions.
  cache.ClearCache();
  Config another_config = CreateDummyCublasLtConfig();
  TF_ASSERT_OK(cache.Insert(instr_1.get(), another_config));
  TF_ASSERT_OK(cache.Insert(instr_2.get(), another_config));

  // Deserialize the cache, only instr_1 should be overwritten.
  TF_ASSERT_OK(cache.Deserialize(serialized_cache));
  EXPECT_THAT(cache.Lookup(instr_1.get()), Optional(ConfigEq(orig_config)));
  EXPECT_THAT(cache.Lookup(instr_2.get()), Optional(ConfigEq(another_config)));
}

TEST_F(LegacyCacheTest, CacheStats) {
  auto cache = LegacyCache(test_dir_, mode_, device_desc_);
  auto instr1 = CreateDummyInstr("hlo_stats1");
  auto instr2 = CreateDummyInstr("hlo_stats2");
  Config config = CreateDummyTritonConfig();

  // Initial stats: 0 hits, 0 misses.
  EXPECT_EQ(cache.GetCacheStats().hits, 0);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  // Lookup miss.
  EXPECT_THAT(cache.Lookup(instr1.get()), Eq(std::nullopt));
  EXPECT_EQ(cache.GetCacheStats().hits, 0);
  EXPECT_EQ(cache.GetCacheStats().misses, 1);

  // Insert and lookup hit.
  TF_ASSERT_OK(cache.Insert(instr1.get(), config));
  EXPECT_THAT(cache.Lookup(instr1.get()), Optional(ConfigEq(config)));
  EXPECT_EQ(cache.GetCacheStats().hits, 1);
  EXPECT_EQ(cache.GetCacheStats().misses, 1);

  // Lookup same instruction again, hit.
  EXPECT_THAT(cache.Lookup(instr1.get()), Optional(ConfigEq(config)));
  EXPECT_EQ(cache.GetCacheStats().hits, 2);
  EXPECT_EQ(cache.GetCacheStats().misses, 1);

  // Lookup another instruction, miss.
  EXPECT_THAT(cache.Lookup(instr2.get()), Eq(std::nullopt));
  EXPECT_EQ(cache.GetCacheStats().hits, 2);
  EXPECT_EQ(cache.GetCacheStats().misses, 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
