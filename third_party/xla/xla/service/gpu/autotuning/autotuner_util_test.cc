/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/autotuner_util.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash_testing.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Ne;
using ::testing::Not;
using ::testing::TempDir;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class AutotunerUtilTest : public HloTestBase {
 protected:
  static constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  static constexpr absl::string_view kResultText = R"(
version: 3
results {
  device: "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: 1555 GB/s, L2 cache: 40 MB"
  hlo: "{\n  tmp_0 = f16[1,16,17,3]{3,2,1,0} parameter(0)\n  tmp_1 = f16[16,51]{1,0} bitcast(f16[1,16,17,3]{3,2,1,0} tmp_0)\n  tmp_2 = s8[16,17,3]{2,1,0} parameter(1)\n  tmp_3 = s8[51,16]{0,1} bitcast(s8[16,17,3]{2,1,0} tmp_2)\n  tmp_4 = f16[51,16]{0,1} convert(s8[51,16]{0,1} tmp_3)\n  tmp_5 = f16[16,16]{1,0} dot(f16[16,51]{1,0} tmp_1, f16[51,16]{0,1} tmp_4), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n  ROOT tmp_6 = f16[1,16,16]{2,1,0} bitcast(f16[16,16]{1,0} tmp_5)\n}"
  result {
    run_time {
      nanos: 31744
    }
    triton {
      block_m: 32
      block_n: 32
      block_k: 32
      split_k: 1
      num_stages: 1
      num_warps: 4
      num_ctas: 1
    }
  }
})";

  void SetUp() override {
    AutotunerUtil::ClearAutotuneResults();
    AutotunerUtil::ClearCacheStats();
  }

  std::string GetUniqueTempFilePath(absl::string_view suffix) {
    std::string filename = TempDir();
    CHECK(tsl::Env::Default()->CreateUniqueFileName(&filename,
                                                    std::string(suffix)));
    return filename;
  }

  std::string ExpectToReadNonEmptyFile(absl::string_view file_path) {
    std::string str;
    tsl::Env* env = tsl::Env::Default();
    TF_EXPECT_OK(tsl::ReadFileToString(env, std::string(file_path), &str));
    EXPECT_THAT(str, Not(IsEmpty()));
    return str;
  }

  static stream_executor::StreamExecutor* NewStreamExecutor() {
    stream_executor::Platform* platform =
        stream_executor::PlatformManager::PlatformWithName("Host").value();
    return platform->ExecutorForDevice(/*ordinal=*/0).value();
  }

  absl::Status PopulateResultCache() {
    EXPECT_TRUE(AutotunerUtil::ResultCacheIsEmpty());
    TF_RETURN_IF_ERROR(AutotunerUtil::LoadAutotuneResults(kResultText, true));
    EXPECT_FALSE(AutotunerUtil::ResultCacheIsEmpty());
    return absl::OkStatus();
  }
};

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_TextProto1) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
  EXPECT_GT(results.results_size(), 0);
}

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_TextProto2) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_Protobuf) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(results.ParseFromString(autotune_results_str));
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_TextProto1) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerUtil::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerUtil::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerUtil::ResultCacheIsEmpty());

  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      std::string(kResultText), &results));
  ASSERT_GT(results.results().size(), 0);
  AddVersionToAutotuneResults(results);
  AutotuneCacheKey key(results.results(0).device(), results.results(0).hlo(),
                       results.results(0).version());
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);
  stream_executor::StreamExecutor* executor = NewStreamExecutor();
  AutotuneConfig config(DeviceConfig{executor}, options);

  EXPECT_THAT(AutotunerUtil::IsInCache(key, config), IsOkAndHolds(true));
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_TextProto2) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerUtil::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerUtil::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerUtil::ResultCacheIsEmpty());
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_Protobuf) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerUtil::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerUtil::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerUtil::ResultCacheIsEmpty());
}

TEST_F(AutotunerUtilTest, ResultConflictsAreDetected) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));
  EXPECT_THAT(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Duplicate autotuning result")));
}

// Test that when complete AOT autotuning is required, and there is cache miss,
// a `NotFound` error will be raised.
TEST_F(AutotunerUtilTest, FailIfRequireCompleteAotAutotuning) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  auto hlo_module = GetOptimizedModule(kHloText);
  TF_EXPECT_OK(hlo_module.status());
  std::vector<HloComputation*> computations =
      (*hlo_module)
          ->MakeNonfusionComputations(absl::flat_hash_set<absl::string_view>());
  EXPECT_THAT(computations, Not(IsEmpty()));
  const HloInstruction* instruction = *computations[0]->instructions().begin();
  stream_executor::StreamExecutor* executor = NewStreamExecutor();
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);
  AutotuneConfig config(DeviceConfig{executor}, options);
  absl::Status s = AutotunerUtil::Autotune(instruction, config, [&] {
                     return AutotuneResult();
                   }).status();
  EXPECT_THAT(
      s, StatusIs(
             absl::StatusCode::kNotFound,
             HasSubstr("Complete XLA AOT autotuning results are required, but "
                       "no AOT result was found for key: <key model")));
  EXPECT_THAT(s.GetPayload(kAutotuneCacheRequiredErrorPayloadKey),
              Ne(std::nullopt));
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 0);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 1);
}

// Test that when JIT autotuning is disabled, but no cache miss due to AOT
// autotuning, `Autotune` still returns Ok status.
TEST_F(AutotunerUtilTest, OkIfJitAutotuningDisabledButAlreadyLoadedAOT) {
  auto hlo_module = GetOptimizedModule(kHloText);
  std::vector<HloComputation*> computations =
      (*hlo_module)
          ->MakeNonfusionComputations(absl::flat_hash_set<absl::string_view>());
  EXPECT_THAT(computations, Not(IsEmpty()));
  const HloInstruction* instruction = *computations[0]->instructions().begin();
  stream_executor::StreamExecutor* executor = NewStreamExecutor();

  {
    // By default, JIT autotuning is OK.
    AutotuneConfig config(DeviceConfig{executor}, DebugOptions());
    TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                   return AutotuneResult();
                 }).status());
    EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 0);
    EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 1);
  }

  // Now require complete AOT autotuning results.
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);

  AutotuneConfig config(DeviceConfig{executor}, options);
  // Even though JIT autotuning is disabled, there is no cache miss when running
  // autotuning for the same entry, so no error should be raised either.
  TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                 return AutotuneResult();
               }).status());
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 1);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 1);
}

class FileBasedCacheTest : public AutotunerUtilTest {
 public:
  static std::string ToString(const AutotuneResult& message) {
    std::string textproto;
    CHECK(tsl::protobuf::TextFormat::PrintToString(message, &textproto));
    return textproto;
  }

  static std::vector<std::string> GetFilesInDir(
      const absl::string_view cache_dir) {
    std::vector<std::string> files_in_cache;
    // TSL's different platform implementations of `GetChildren` are not
    // consistent. Some return an error if `cache_dir` does not exist, some
    // others return an empty `files_in_cache`. We want the second behavior, so
    // we swallow the error.
    if (!tsl::Env::Default()
             ->GetChildren(std::string(cache_dir), &files_in_cache)
             .ok()) {
      files_in_cache.clear();
    }
    return files_in_cache;
  }

  static std::string Read(const absl::string_view filepath) {
    std::string file_content;
    TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(),
                                      std::string(filepath), &file_content));
    return file_content;
  }

  void Write(const absl::string_view filepath,
             const absl::string_view content) {
    TF_CHECK_OK(CreateDirIfNeeded(cache_dir_, tsl::Env::Default()));
    TF_CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(),
                                       std::string(filepath), content));
  }

  stream_executor::StreamExecutor* executor_ = NewStreamExecutor();
  std::unique_ptr<HloModule> module_ =
      ParseAndReturnVerifiedModule(kHloText).value();
  const HloInstruction* dot_ = hlo_query::GetFirstInstructionWithOpcode(
      *module_->entry_computation(), HloOpcode::kDot);
  std::string cache_dir_ = [] {
    tsl::Env* default_env = tsl::Env::Default();
    std::string cache_dir;
    CHECK(default_env->LocalTempFilename(&cache_dir));
    return cache_dir;
  }();

  DebugOptions::AutotuneCacheMode GetCacheMode() const { return cache_mode_; }
  void SetCacheMode(DebugOptions::AutotuneCacheMode cache_mode) {
    cache_mode_ = cache_mode;
  }

  AutotuneConfig GetConfig() const {
    DebugOptions options;
    options.set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir_);
    options.set_xla_gpu_experimental_autotune_cache_mode(GetCacheMode());
    return AutotuneConfig(DeviceConfig{executor_}, options);
  }

  AutotuneCacheKey GetCacheKey() const {
    return AutotunerUtil::GetKey(dot_, GetConfig());
  }

  std::string GetCacheFilename() const {
    absl::StatusOr<std::string> key_hash =
        GetBase64EncodedSha256Hash(GetCacheKey().ToString());
    CHECK_OK(key_hash.status());
    return absl::StrCat(key_hash.value(), ".textproto");
  }

  std::string GetCacheFilePath() const {
    return tsl::io::JoinPath(cache_dir_, GetCacheFilename());
  }
  const AutotuneResult result1_ = [] {
    AutotuneResult result;
    result.set_scratch_bytes(1);
    return result;
  }();
  const AutotuneResult result2_ = [] {
    AutotuneResult result;
    result.set_scratch_bytes(2);
    return result;
  }();

 private:
  DebugOptions::AutotuneCacheMode cache_mode_ =
      DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE;
};

TEST_F(FileBasedCacheTest, AutotuneCreatesTmpAndWritesResultToTheCacheDir) {
  TF_ASSERT_OK_AND_ASSIGN(
      AutotuneResult result,
      AutotunerUtil::Autotune(dot_, GetConfig(), [&] { return result1_; }));
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 0);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 1);
  EXPECT_EQ(ToString(result), ToString(result1_));

  ASSERT_THAT(GetFilesInDir(cache_dir_),
              UnorderedElementsAre(GetCacheFilename(), "tmp"));
  EXPECT_EQ(Read(GetCacheFilePath()), ToString(result1_));
}

TEST_F(FileBasedCacheTest, AutotuneReadsResultFromTheCacheDir) {
  Write(GetCacheFilePath(), ToString(result1_));

  bool cache_hit = true;
  TF_ASSERT_OK_AND_ASSIGN(AutotuneResult result,
                          AutotunerUtil::Autotune(dot_, GetConfig(), [&] {
                            cache_hit = false;
                            return result2_;
                          }));

  EXPECT_TRUE(cache_hit);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 1);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 0);
  EXPECT_EQ(ToString(result), ToString(result1_));
}

TEST_F(FileBasedCacheTest,
       RepeatedAutotuneCallsDontReadOrWriteTheCacheFileAgain) {
  auto check_autotune_cache_hit = [](const HloInstruction* instr,
                                     const AutotuneConfig& config,
                                     const AutotuneResult& expected_result) {
    bool cache_hit = true;
    TF_ASSERT_OK_AND_ASSIGN(AutotuneResult result,
                            AutotunerUtil::Autotune(instr, config, [&] {
                              cache_hit = false;
                              AutotuneResult new_result;
                              new_result.set_scratch_bytes(2);
                              return new_result;
                            }));
    EXPECT_TRUE(cache_hit);
    EXPECT_EQ(ToString(result), ToString(expected_result));
  };
  const std::string cache_file_path = GetCacheFilePath();
  const AutotuneConfig config = GetConfig();

  Write(cache_file_path, ToString(result1_));
  check_autotune_cache_hit(dot_, config, /*expected_result=*/result1_);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 1);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 0);

  constexpr absl::string_view kPlaceholderContent = "placeholder content";
  Write(cache_file_path, kPlaceholderContent);
  // File was not read again:
  check_autotune_cache_hit(dot_, config, /*expected_result=*/result1_);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 2);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 0);
  // File was not written again:
  EXPECT_EQ(Read(cache_file_path), kPlaceholderContent);
}

TEST_F(FileBasedCacheTest,
       IsInCacheReturnsTrueIfTheResultIsInTheFileBasedCache) {
  Write(GetCacheFilePath(), ToString(result1_));

  TF_ASSERT_OK_AND_ASSIGN(bool is_in_cache,
                          AutotunerUtil::IsInCache(GetCacheKey(), GetConfig()));

  EXPECT_TRUE(is_in_cache);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 1);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 0);
}

TEST_F(FileBasedCacheTest, IsInCacheReturnsFalseIfTheResultIsNotInEitherCache) {
  TF_ASSERT_OK_AND_ASSIGN(bool is_in_cache,
                          AutotunerUtil::IsInCache(GetCacheKey(), GetConfig()));

  EXPECT_FALSE(is_in_cache);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_hits, 0);
  EXPECT_EQ(AutotunerUtil::GetCacheStats().cache_misses, 1);
}

TEST_F(FileBasedCacheTest, AddResultAddsTheResultToTheFileBasedCache) {
  TF_ASSERT_OK_AND_ASSIGN(
      bool added,
      AutotunerUtil::AddResult(GetCacheKey(), result1_, GetConfig()));
  EXPECT_TRUE(added);

  ASSERT_THAT(GetFilesInDir(cache_dir_),
              UnorderedElementsAre(GetCacheFilename(), "tmp"));
  EXPECT_EQ(Read(GetCacheFilePath()), ToString(result1_));
}

TEST_F(FileBasedCacheTest, RepeatedAddResultDoesNotWriteTheFileAgain) {
  const std::string cache_file_path = GetCacheFilePath();
  const AutotuneCacheKey cache_key = GetCacheKey();
  const AutotuneConfig config = GetConfig();
  {
    TF_ASSERT_OK_AND_ASSIGN(
        bool added, AutotunerUtil::AddResult(cache_key, result1_, config));
    EXPECT_TRUE(added);
  }
  ASSERT_THAT(GetFilesInDir(cache_dir_),
              UnorderedElementsAre(GetCacheFilename(), "tmp"));
  EXPECT_EQ(Read(cache_file_path), ToString(result1_));
  constexpr absl::string_view kPlaceholderContent = "placeholder content";
  Write(cache_file_path, kPlaceholderContent);

  {
    TF_ASSERT_OK_AND_ASSIGN(
        bool added, AutotunerUtil::AddResult(cache_key, result1_, config));
    EXPECT_FALSE(added);
  }

  // File was not written again:
  EXPECT_EQ(Read(cache_file_path), kPlaceholderContent);
}

TEST(AutotuneCacheKeyTest, DeviceDescriptionToCacheKey) {
  auto device_description =
      [](absl::string_view spec_file_name) -> se::DeviceDescription {
    se::GpuTargetConfigProto proto;
    std::string spec_string;
    CHECK_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools", "hlo_opt",
                          "gpu_specs", spec_file_name),
        &spec_string));
    EXPECT_TRUE(
        tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto));
    return se::DeviceDescription(proto.gpu_device_info());
  };

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("a100_sxm_40.txtpb")),
            "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: "
            "1555 GB/s, L2 cache: 40 MB");

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("a100_sxm_80.txtpb")),
            "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: "
            "2039 GB/s, L2 cache: 40 MB");

  EXPECT_EQ(AutotuneCacheKey::DeviceDescriptionToCacheKey(
                device_description("mi200.txtpb")),
            "ROCM: gfx90a, Cores: 110, GPU clock: 1.7 GHz, Memory bandwidth: "
            "1638 GB/s, L2 cache: 8 MB");
}

TEST(AutotuneCacheKeyTest, VersionIsIncludedInCacheKey) {
  AutotuneCacheKey key = AutotuneCacheKey("model", "hlo");
  EXPECT_THAT(key.ToString(),
              HasSubstr(absl::StrFormat("version=%d", key.GetVersion())));
}

TEST(AutotuneCacheKeyTest, VersionChangeInvalidateCacheKey) {
  AutotuneCacheKey key0 = AutotuneCacheKey("model", "hlo", /*version=*/0);
  AutotuneCacheKey key1 = AutotuneCacheKey("model", "hlo", /*version=*/1);
  EXPECT_FALSE(key0 == key1);
  EXPECT_NE(key0.ToString(), key1.ToString());
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      key0,
      key1,
  }));
}

TEST_F(FileBasedCacheTest, AddResultDoesNotWriteTheFileInReadMode) {
  SetCacheMode(DebugOptions::AUTOTUNE_CACHE_MODE_READ);
  TF_ASSERT_OK_AND_ASSIGN(
      bool added,
      AutotunerUtil::AddResult(GetCacheKey(), result1_, GetConfig()));
  EXPECT_TRUE(added);  // was added to in memory cache.
  EXPECT_EQ(GetFilesInDir(cache_dir_).size(),
            0);  // wasn't dumped to file based cache.
}

}  // namespace
}  // namespace gpu
}  // namespace xla
