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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
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
}  // namespace

class LocalCacheTest : public HloHardwareIndependentTestBase {
 protected:
  AutotunerCacheInterface::Config CreateTestConfig(
      autotuner::Backend backend = autotuner::Backend::TRITON) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    return cfg;
  }
};

TEST_F(LocalCacheTest, BasicInsertAndLookup) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* instr1 =
      module->entry_computation()->root_instruction();

  LocalCacheStorage storage;
  LocalCache cache(KeyMatchingMode::kStrict, &storage);
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

  LocalCacheStorage storage;
  LocalCache cache(KeyMatchingMode::kLoose, &storage);

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

  LocalCacheStorage storage;
  LocalCache cache(KeyMatchingMode::kStrict, &storage);
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

  LocalCacheStorage storage_src;
  LocalCache cache_src(KeyMatchingMode::kStrict, &storage_src);
  AutotunerCacheInterface::Config config1 =
      CreateTestConfig(autotuner::Backend::TRITON);
  EXPECT_OK(cache_src.Insert(dot, config1));

  // Serialize all entries
  ASSERT_OK_AND_ASSIGN(std::string serialized, cache_src.Serialize({}));

  // Deserialize into a second cache
  LocalCacheStorage storage_dest;
  LocalCache cache_dest(KeyMatchingMode::kStrict, &storage_dest);
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

  LocalCacheStorage storage_src;
  LocalCache cache_src(KeyMatchingMode::kStrict, &storage_src);
  EXPECT_OK(
      cache_src.Insert(dot, CreateTestConfig(autotuner::Backend::TRITON)));
  EXPECT_OK(
      cache_src.Insert(operand, CreateTestConfig(autotuner::Backend::TRITON)));

  // Serialize only the dot instruction
  ASSERT_OK_AND_ASSIGN(std::string serialized, cache_src.Serialize({dot}));

  // Deserialize into a second cache
  LocalCacheStorage storage_dest;
  LocalCache cache_dest(KeyMatchingMode::kStrict, &storage_dest);
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

}  // namespace xla
