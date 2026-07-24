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

#include "xla/backends/autotuner/local_cache.h"

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
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"

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
}  // namespace

class LocalCacheTest : public HloHardwareIndependentTestBase {
 protected:
  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }

  LocalCacheStorage storage_;
};

TEST_F(LocalCacheTest, BasicInsertAndLookup) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  LocalCache cache(CreateCacheContext(), KeyMatchingMode::kStrict, &storage_);
  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(instr1, config));

  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(instr1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  EXPECT_EQ(cache.GetCacheStats().hits, 1);
  EXPECT_EQ(cache.GetCacheStats().misses, 0);

  const HloInstruction* instr2 = instr1->operand(0);
  std::optional<AutotunerCacheInterface::Config> result_miss =
      cache.Lookup(instr2);
  EXPECT_FALSE(result_miss.has_value());
  EXPECT_EQ(cache.GetCacheStats().misses, 1);
}

TEST_F(LocalCacheTest, LooseMatchingMode) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCache cache(CreateCacheContext(), KeyMatchingMode::kLoose, &storage_);

  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(dot, config));

  // In Loose mode, looking up with different codegen options should still hit.
  dot->GetModule()
      ->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_cublas_fallback(true);
  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);
}

TEST_F(LocalCacheTest, StrictMatchingMode) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCache cache(CreateCacheContext(), KeyMatchingMode::kStrict, &storage_);
  AutotunerCacheInterface::Config config = CreateTestConfig();

  EXPECT_OK(cache.Insert(dot, config));

  // In Strict mode, looking up with different codegen options should miss.
  bool current_val =
      dot->GetModule()->config().debug_options().xla_gpu_cublas_fallback();
  dot->GetModule()
      ->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_cublas_fallback(!current_val);
  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(dot);
  EXPECT_FALSE(result.has_value());
}

TEST_F(LocalCacheTest, SerializeAndDeserializeAll) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  std::string serialized;
  LocalCacheStorage storage_dest;
  {
    LocalCache cache_src(CreateCacheContext(), KeyMatchingMode::kStrict,
                         &storage_);
    AutotunerCacheInterface::Config config1 =
        CreateTestConfig(autotuner::Backend::TRITON);
    EXPECT_OK(cache_src.Insert(dot, config1));

    // Serialize all entries
    ASSERT_OK_AND_ASSIGN(serialized, cache_src.Serialize({}));
  }

  // Deserialize into a second cache
  LocalCache cache_dest(CreateCacheContext(), KeyMatchingMode::kStrict,
                        &storage_dest);
  EXPECT_OK(cache_dest.Deserialize(serialized));

  // Verify lookup has the entry
  std::optional<AutotunerCacheInterface::Config> result =
      cache_dest.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);
}

TEST_F(LocalCacheTest, SerializeInstructionsSubset) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  const HloInstruction* operand = dot->operand(0);

  std::string serialized;
  LocalCacheStorage storage_dest;
  {
    LocalCache cache_src(CreateCacheContext(), KeyMatchingMode::kStrict,
                         &storage_);
    EXPECT_OK(
        cache_src.Insert(dot, CreateTestConfig(autotuner::Backend::TRITON)));
    EXPECT_OK(cache_src.Insert(operand,
                               CreateTestConfig(autotuner::Backend::TRITON)));

    // Serialize only the dot instruction
    ASSERT_OK_AND_ASSIGN(serialized, cache_src.Serialize({dot}));
  }

  // Deserialize into a second cache
  LocalCache cache_dest(CreateCacheContext(), KeyMatchingMode::kStrict,
                        &storage_dest);
  EXPECT_OK(cache_dest.Deserialize(serialized));

  // Verify dot is present but operand is missing
  std::optional<AutotunerCacheInterface::Config> result_dot =
      cache_dest.Lookup(dot);
  ASSERT_TRUE(result_dot.has_value());
  EXPECT_EQ(result_dot->codegen_backend, autotuner::Backend::TRITON);

  std::optional<AutotunerCacheInterface::Config> result_operand =
      cache_dest.Lookup(operand);
  EXPECT_FALSE(result_operand.has_value());
}

TEST_F(LocalCacheTest, DeserializeSkipsDeviceMismatch) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage storage_dest;
  LocalCache cache_src(CreateCacheContext("v1.0", 108),
                       KeyMatchingMode::kStrict, &storage_);
  EXPECT_OK(
      cache_src.Insert(dot, CreateTestConfig(autotuner::Backend::TRITON)));

  ASSERT_OK_AND_ASSIGN(std::string serialized, cache_src.Serialize({}));

  LocalCache cache_dest(CreateCacheContext("v1.0", 80),
                        KeyMatchingMode::kStrict, &storage_dest);

  EXPECT_OK(cache_dest.Deserialize(serialized));
  std::optional<AutotunerCacheInterface::Config> result =
      cache_dest.Lookup(dot);
  EXPECT_FALSE(result.has_value());
}

TEST_F(LocalCacheTest, DeserializeSkipsCodegenMismatchStrict) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage storage_dest;
  LocalCache cache_src(
      CreateCacheContext("v1.0", 108,
                         {{autotuner::Backend::TRITON, "triton_v1"}}),
      KeyMatchingMode::kStrict, &storage_);
  EXPECT_OK(
      cache_src.Insert(dot, CreateTestConfig(autotuner::Backend::TRITON)));

  ASSERT_OK_AND_ASSIGN(std::string serialized, cache_src.Serialize({}));

  // Change backend version to change codegen_version
  LocalCache cache_dest(
      CreateCacheContext("v1.0", 108,
                         {{autotuner::Backend::TRITON, "triton_v2"}}),
      KeyMatchingMode::kStrict, &storage_dest);

  EXPECT_OK(cache_dest.Deserialize(serialized));
  std::optional<AutotunerCacheInterface::Config> result =
      cache_dest.Lookup(dot);
  EXPECT_FALSE(result.has_value());
}

}  // namespace xla
