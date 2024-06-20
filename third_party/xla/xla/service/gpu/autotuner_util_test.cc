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

#include "xla/service/gpu/autotuner_util.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::TempDir;
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
  device: "sm_8.0 with 42331013120B RAM, 108 cores, 1410000KHz clock, 1215000KHz mem clock, 41943040B L2$"
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

  void SetUp() override { AutotunerUtil::ClearAutotuneResults(); }

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

  static std::unique_ptr<stream_executor::StreamExecutor> NewStreamExecutor() {
    stream_executor::Platform* platform =
        stream_executor::PlatformManager::PlatformWithName("Host").value();
    stream_executor::StreamExecutorConfig config(/*ordinal=*/0);
    return platform->GetUncachedExecutor(config).value();
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
  std::unique_ptr<stream_executor::StreamExecutor> executor =
      NewStreamExecutor();
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);
  AutotuneConfig config(DeviceConfig{executor.get()}, options);
  EXPECT_THAT(
      AutotunerUtil::Autotune(instruction, config,
                              [&] { return AutotuneResult(); }),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Complete XLA AOT autotuning results are required, but "
                    "no AOT result was found for key: <key model")));
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
  std::unique_ptr<stream_executor::StreamExecutor> executor =
      NewStreamExecutor();

  {
    // By default, JIT autotuning is OK.
    AutotuneConfig config(DeviceConfig{executor.get()}, DebugOptions());
    TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                   return AutotuneResult();
                 }).status());
  }

  // Now require complete AOT autotuning results.
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);

  AutotuneConfig config(DeviceConfig{executor.get()}, options);
  // Even though JIT autotuning is disabled, there is no cache miss when running
  // autotuning for the same entry, so no error should be raised either.
  TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                 return AutotuneResult();
               }).status());
}

class MockExternalCache : public ExternalCacheInterface {
 public:
  MOCK_METHOD(absl::StatusOr<std::optional<std::string>>, Get,
              (ExternalCacheItemType type, absl::string_view key), (override));
  MOCK_METHOD(absl::Status, Set,
              (ExternalCacheItemType type, absl::string_view key,
               absl::string_view value),
              (override));
};

class ExternalCacheUsageTest : public AutotunerUtilTest {
 public:
  std::unique_ptr<stream_executor::StreamExecutor> executor_ =
      NewStreamExecutor();
  std::unique_ptr<HloModule> module_ =
      ParseAndReturnVerifiedModule(kHloText).value();
  const HloInstruction* dot_ = hlo_query::GetFirstInstructionWithOpcode(
      *module_->entry_computation(), HloOpcode::kDot);
  std::shared_ptr<StrictMock<MockExternalCache>> mock_external_cache_ =
      std::make_shared<StrictMock<MockExternalCache>>();
  AutotuneConfig config_ =
      AutotuneConfig(DeviceConfig{executor_.get()}, mock_external_cache_);
  AutotuneCacheKey cache_key_ = AutotunerUtil::GetKey(dot_, config_);
  const AutotuneResult result_ = [] {
    AutotuneResult result;
    result.set_scratch_bytes(1);
    return result;
  }();
};

std::string ToString(const proto2::Message& message) {
  std::string textproto;
  CHECK(tsl::protobuf::TextFormat::PrintToString(message, &textproto));
  return textproto;
}

TEST_F(ExternalCacheUsageTest, AutotuneReadsResultFromExternalCacheIfFound) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(ToString(result_)));

  TF_ASSERT_OK_AND_ASSIGN(
      AutotuneResult result,
      AutotunerUtil::Autotune(dot_, config_, [&] { return result_; }));
  EXPECT_EQ(ToString(result), ToString(result_));
}

TEST_F(ExternalCacheUsageTest, AutotuneStoresResultToExternalCacheIfMissing) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(std::nullopt));
  EXPECT_CALL(*mock_external_cache_,
              Set(ExternalCacheItemType::kAutotuneResult, cache_key_.ToString(),
                  ToString(result_)))
      .WillOnce(Return(absl::OkStatus()));

  TF_ASSERT_OK_AND_ASSIGN(
      AutotuneResult result,
      AutotunerUtil::Autotune(dot_, config_, [&] { return result_; }));
  EXPECT_EQ(ToString(result), ToString(result_));
}

TEST_F(ExternalCacheUsageTest, RepeatedAutotuneCallsDontUseTheExternalCache) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(ToString(result_)));

  for (int i = 0; i < 2; i++) {
    TF_ASSERT_OK_AND_ASSIGN(AutotuneResult result,
                            AutotunerUtil::Autotune(dot_, config_, [&] {
                              EXPECT_FALSE(true) << "should not be called";
                              return AutotuneResult();
                            }));
    EXPECT_EQ(ToString(result), ToString(result_));
  }
}

TEST_F(ExternalCacheUsageTest,
       IsInCacheReturnsTrueIfTheResultIsInTheExternalCache) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(ToString(result_)));

  TF_ASSERT_OK_AND_ASSIGN(bool is_in_cache,
                          AutotunerUtil::IsInCache(cache_key_, config_));

  EXPECT_TRUE(is_in_cache);
}

TEST_F(ExternalCacheUsageTest,
       IsInCacheReturnsFalseIfTheResultIsNotInEitherCaches) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(bool is_in_cache,
                          AutotunerUtil::IsInCache(cache_key_, config_));

  EXPECT_FALSE(is_in_cache);
}

TEST_F(ExternalCacheUsageTest,
       RepeatedIsInCacheCallsCheckTheExternalCacheUntilTheElementIsFound) {
  EXPECT_CALL(*mock_external_cache_, Get(ExternalCacheItemType::kAutotuneResult,
                                         cache_key_.ToString()))
      .WillOnce(Return(std::nullopt))
      .WillOnce(Return(std::nullopt))
      .WillOnce(Return(ToString(result_)));

  bool is_in_cache = false;
  for (int i = 0; i < 2; i++) {
    TF_ASSERT_OK_AND_ASSIGN(is_in_cache,
                            AutotunerUtil::IsInCache(cache_key_, config_));
    EXPECT_FALSE(is_in_cache);
  }
  for (int i = 0; i < 2; i++) {
    TF_ASSERT_OK_AND_ASSIGN(is_in_cache,
                            AutotunerUtil::IsInCache(cache_key_, config_));
    EXPECT_TRUE(is_in_cache);
  }
}

TEST_F(ExternalCacheUsageTest, AddResultAddsTheResultToTheExternalCache) {
  EXPECT_CALL(*mock_external_cache_,
              Set(ExternalCacheItemType::kAutotuneResult, cache_key_.ToString(),
                  ToString(result_)))
      .WillOnce(Return(absl::OkStatus()));

  TF_ASSERT_OK_AND_ASSIGN(
      bool added, AutotunerUtil::AddResult(cache_key_, result_, config_));
  EXPECT_TRUE(added);
}

TEST_F(ExternalCacheUsageTest,
       RepeatedAddResultDoesNotAddTheResultToTheExternalCacheAgain) {
  EXPECT_CALL(*mock_external_cache_,
              Set(ExternalCacheItemType::kAutotuneResult, cache_key_.ToString(),
                  ToString(result_)))
      .WillOnce(Return(absl::OkStatus()));

  bool added = false;
  TF_ASSERT_OK_AND_ASSIGN(
      added, AutotunerUtil::AddResult(cache_key_, result_, config_));
  EXPECT_TRUE(added);
  TF_ASSERT_OK_AND_ASSIGN(
      added, AutotunerUtil::AddResult(cache_key_, result_, config_));
  EXPECT_FALSE(added);
}

std::string Read(const absl::string_view filepath) {
  std::string file_content;
  TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), std::string(filepath),
                                    &file_content));
  return file_content;
}

void Write(const absl::string_view filepath, const absl::string_view content) {
  TF_CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), std::string(filepath),
                                     content));
}

std::string CreateEmptyDir() {
  tsl::Env* default_env = tsl::Env::Default();
  std::string dir;
  CHECK(default_env->LocalTempFilename(&dir));
  CHECK_OK(default_env->CreateDir(dir));
  return dir;
}

class AutotuneConfigTest : public AutotunerUtilTest {};

TEST_F(AutotuneConfigTest,
       SetsExternalCacheToNullIfNoDirectoryIsProvidedInDebugOptions) {
  AutotuneConfig config(DeviceConfig{NewStreamExecutor().get()},
                        DebugOptions());
  EXPECT_EQ(config.external_cache(), nullptr);
}

TEST_F(AutotuneConfigTest,
       SetsExternalCacheToFileBasedCacheUsingTheProvidedDirectory) {
  constexpr absl::string_view kCacheDir = "dir";
  DebugOptions debug_options;
  debug_options.set_xla_gpu_per_fusion_autotune_cache_dir(kCacheDir);
  AutotuneConfig config(DeviceConfig{NewStreamExecutor().get()}, debug_options);

  FileBasedCache* cache =
      dynamic_cast<FileBasedCache*>(config.external_cache());
  ASSERT_NE(cache, nullptr);
  EXPECT_EQ(cache->cache_dir(), kCacheDir);
}

TEST(FileBasedCacheTest, GetReturnsContentOfFileIfFound) {
  constexpr absl::string_view kKey = "key";
  constexpr absl::string_view kValue = "value";
  const std::string cache_dir = CreateEmptyDir();
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string file_path,
      FileBasedCache::GetCacheFilePath(
          cache_dir, ExternalCacheItemType::kAutotuneResult, kKey));
  Write(file_path, kValue);
  FileBasedCache cache(cache_dir);

  TF_ASSERT_OK_AND_ASSIGN(
      std::optional<std::string> result,
      cache.Get(ExternalCacheItemType::kAutotuneResult, kKey));

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), kValue);
}

TEST(FileBasedCacheTest, GetReturnsNulloptIfNotFound) {
  FileBasedCache cache(CreateEmptyDir());

  TF_ASSERT_OK_AND_ASSIGN(
      std::optional<std::string> result,
      cache.Get(ExternalCacheItemType::kAutotuneResult, "key"));

  ASSERT_FALSE(result.has_value());
}

TEST(FileBasedCacheTest, SetWritesValueToFile) {
  constexpr absl::string_view kKey = "key";
  constexpr absl::string_view kValue = "value";

  std::string cache_dir = CreateEmptyDir();
  FileBasedCache cache(cache_dir);

  ASSERT_OK(cache.Set(ExternalCacheItemType::kAutotuneResult, kKey, kValue));

  TF_ASSERT_OK_AND_ASSIGN(
      const std::string file_path,
      FileBasedCache::GetCacheFilePath(
          cache_dir, ExternalCacheItemType::kAutotuneResult, kKey));
  EXPECT_EQ(Read(file_path), kValue);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
