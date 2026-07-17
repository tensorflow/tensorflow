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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/fingerprint.h"

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
      autotuner::Backend backend = autotuner::Backend::TRITON,
      int64_t block_m = 0) {
    AutotunerCacheInterface::Config cfg;
    cfg.codegen_backend = backend;
    if (backend == autotuner::Backend::TRITON && block_m != 0) {
      cfg.backend_config.mutable_triton()->set_block_m(block_m);
    }
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

TEST_F(LocalCacheTest, ProcessWideStorageSharing) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage::GetInstance().Clear();

  LocalCache cache1(CreateCacheContext(), KeyMatchingMode::kStrict);
  LocalCache cache2(CreateCacheContext(), KeyMatchingMode::kStrict);

  AutotunerCacheInterface::Config config = CreateTestConfig();
  EXPECT_OK(cache1.Insert(dot, config));

  std::optional<AutotunerCacheInterface::Config> result = cache2.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);

  EXPECT_EQ(cache1.GetCacheStats().hits, 1);
  EXPECT_EQ(cache2.GetCacheStats().hits, 1);
}

TEST_F(LocalCacheTest, LooseMatchingCrossCompatibility) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage::GetInstance().Clear();

  AutotuneCacheContext context_a =
      CreateCacheContext("v1.0", 108,
                         {{autotuner::Backend::TRITON, "triton_v1"},
                          {autotuner::Backend::CUDNN, "cudnn_v1"}});
  AutotuneCacheContext context_b = CreateCacheContext(
      "v1.0", 108, {{autotuner::Backend::TRITON, "triton_v1"}});

  LocalCache cache_a(context_a, KeyMatchingMode::kStrict);
  LocalCache cache_b_strict(context_b, KeyMatchingMode::kStrict);
  LocalCache cache_b_loose(context_b, KeyMatchingMode::kLoose);

  AutotunerCacheInterface::Config config =
      CreateTestConfig(autotuner::Backend::TRITON);
  EXPECT_OK(cache_a.Insert(dot, config));

  std::optional<AutotunerCacheInterface::Config> result_strict =
      cache_b_strict.Lookup(dot);
  EXPECT_FALSE(result_strict.has_value());

  std::optional<AutotunerCacheInterface::Config> result_loose =
      cache_b_loose.Lookup(dot);
  ASSERT_TRUE(result_loose.has_value());
  EXPECT_EQ(result_loose->codegen_backend, autotuner::Backend::TRITON);
}

TEST_F(LocalCacheTest, TextprotoDeserialization) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage::GetInstance().Clear();

  tsl::Fprint128 hlo_fp = GetHloFingerprint(*dot);
  std::string hlo_fp_str =
      absl::StrCat(absl::Hex(hlo_fp.high64, absl::kZeroPad16),
                   absl::Hex(hlo_fp.low64, absl::kZeroPad16));

  AutotuneCacheContext context = CreateCacheContext();

  std::string textproto =
      absl::StrFormat(R"pb(
                        device_scope: "%s"
                        explicit_version_scope: "v1.0"
                        entries {
                          key {
                            target {
                              hlo_fingerprint: "%s"
                              device: "%s"
                              explicit_version: "v1.0"
                            }
                            environment { codegen_version: "fake_env" }
                          }
                          value {
                            optimal_config {
                              backend: TRITON
                              backend_config { triton { block_m: 12345 } }
                            }
                            optimal_backend_version: "triton_v1"
                          }
                        }
                      )pb",
                      context.device(), hlo_fp_str, context.device());

  LocalCache cache(context, KeyMatchingMode::kLoose);
  EXPECT_OK(cache.Deserialize(textproto));

  std::optional<AutotunerCacheInterface::Config> result = cache.Lookup(dot);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->codegen_backend, autotuner::Backend::TRITON);
  EXPECT_EQ(result->backend_config.triton().block_m(), 12345);
}

TEST_F(LocalCacheTest, DeserializationRetainsAllEntries) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo1));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  LocalCacheStorage::GetInstance().Clear();

  tsl::Fprint128 hlo_fp = GetHloFingerprint(*dot);
  std::string hlo_fp_str =
      absl::StrCat(absl::Hex(hlo_fp.high64, absl::kZeroPad16),
                   absl::Hex(hlo_fp.low64, absl::kZeroPad16));

  AutotuneCacheContext context_a = CreateCacheContext(
      "v1.0", 108, {{autotuner::Backend::TRITON, "triton_v1"}}, "device_A");
  AutotuneCacheContext context_b = CreateCacheContext(
      "v2.0", 108, {{autotuner::Backend::TRITON, "triton_v1"}}, "device_B");

  std::string textproto = absl::StrFormat(
      R"pb(
        entries {
          key {
            target {
              hlo_fingerprint: "%s"
              device: "%s"
              explicit_version: "v1.0"
            }
            environment { codegen_version: "fake_env" }
          }
          value {
            optimal_config {
              backend: TRITON
              backend_config { triton { block_m: 100 } }
            }
            optimal_backend_version: "triton_v1"
          }
        }
        entries {
          key {
            target {
              hlo_fingerprint: "%s"
              device: "%s"
              explicit_version: "v2.0"
            }
            environment { codegen_version: "fake_env" }
          }
          value {
            optimal_config {
              backend: TRITON
              backend_config { triton { block_m: 200 } }
            }
            optimal_backend_version: "triton_v1"
          }
        }
      )pb",
      hlo_fp_str, context_a.device(), hlo_fp_str, context_b.device());

  LocalCache cache_a(context_a, KeyMatchingMode::kLoose);
  EXPECT_OK(cache_a.Deserialize(textproto));

  std::optional<AutotunerCacheInterface::Config> result_a = cache_a.Lookup(dot);
  ASSERT_TRUE(result_a.has_value());
  EXPECT_EQ(result_a->backend_config.triton().block_m(), 100);

  LocalCache cache_b(context_b, KeyMatchingMode::kLoose);
  std::optional<AutotunerCacheInterface::Config> result_b = cache_b.Lookup(dot);
  ASSERT_TRUE(result_b.has_value());
  EXPECT_EQ(result_b->backend_config.triton().block_m(), 200);
}

}  // namespace xla
