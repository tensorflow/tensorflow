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

#include "xla/service/gpu/autotuning/autotuner_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/autotuning/autotune_cache_key.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Optional;
using ::testing::TempDir;
using ::testing::UnorderedElementsAre;

static constexpr absl::string_view kDeviceDescriptionTextProto = R"pb(
  core_count: 108
  clock_rate_ghz: 1.41
  memory_bandwidth: 1555000000000
  l2_cache_size: 41943040
  cuda_compute_capability { major: 8 }
)pb";

static constexpr absl::string_view kDotFusionHloText = R"hlo(
    HloModule module
    fused_computation {
          tmp_0 = f16[1,16,17,3]{3,2,1,0} parameter(0)
          tmp_1 = f16[16,51]{1,0} bitcast(f16[1,16,17,3]{3,2,1,0} tmp_0)
          tmp_2 = s8[16,17,3]{2,1,0} parameter(1)
          tmp_3 = s8[51,16]{0,1} bitcast(s8[16,17,3]{2,1,0} tmp_2)
          tmp_4 = f16[51,16]{0,1} convert(s8[51,16]{0,1} tmp_3)
          tmp_5 = f16[16,16]{1,0} dot(f16[16,51]{1,0} tmp_1, f16[51,16]{0,1} tmp_4), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          ROOT tmp_6 = f16[1,16,16]{2,1,0} bitcast(f16[16,16]{1,0} tmp_5)
    }

    ENTRY main {
          p0 = f16[1,16,17,3]{3,2,1,0} parameter(0)
          p1 = s8[16,17,3]{2,1,0} parameter(1)
          ROOT fusion = f16[1,16,16]{2,1,0} fusion(p0, p1), kind=kCustom, calls=fused_computation
    }
  )hlo";

class AutotunerCacheTest : public HloTestBase {
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

  static constexpr absl::string_view kResultText = R"pb(
    version: 3
    results {
      device: "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: 1555 GB/s, L2 cache: 40 MB, DNN version: 1.2.3"
      hlo: "{\n  tmp_0 = f16[1,16,17,3]{3,2,1,0} parameter(0)\n  tmp_1 = f16[16,51]{1,0} bitcast(f16[1,16,17,3]{3,2,1,0} tmp_0)\n  tmp_2 = s8[16,17,3]{2,1,0} parameter(1)\n  tmp_3 = s8[51,16]{0,1} bitcast(s8[16,17,3]{2,1,0} tmp_2)\n  tmp_4 = f16[51,16]{0,1} convert(s8[51,16]{0,1} tmp_3)\n  tmp_5 = f16[16,16]{1,0} dot(f16[16,51]{1,0} tmp_1, f16[51,16]{0,1} tmp_4), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n  ROOT tmp_6 = f16[1,16,16]{2,1,0} bitcast(f16[16,16]{1,0} tmp_5)\n}"
      result {
        run_time { nanos: 31744 }
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
    })pb";

  void SetUp() override {
    AutotunerCache::ClearAutotuneResults();
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
    EXPECT_TRUE(AutotunerCache::ResultCacheIsEmpty());
    TF_RETURN_IF_ERROR(AutotunerCache::LoadAutotuneResults(kResultText, true));
    EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());
    return absl::OkStatus();
  }
};

TEST_F(AutotunerCacheTest, SerializeAutotuneResultsToFile_TextProto1) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
  EXPECT_GT(results.results_size(), 0);
}

TEST_F(AutotunerCacheTest, SerializeAutotuneResultsToFile_TextProto2) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotunerCacheTest, SerializeAutotuneResultsToFile_Protobuf) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(results.ParseFromString(autotune_results_str));
}

TEST_F(AutotunerCacheTest, LoadAutotuneResultsFromFile_TextProto1) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerCache::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerCache::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerCache::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());

  stream_executor::GpuDeviceInfoProto device_description_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      kDeviceDescriptionTextProto, &device_description_proto));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kDotFusionHloText));

  AutotuneResults results;
  EXPECT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(kResultText, &results));
  ASSERT_GT(results.results().size(), 0);
  AddVersionToAutotuneResults(results);
  TF_ASSERT_OK_AND_ASSIGN(
      stream_executor::DeviceDescription device_description,
      stream_executor::DeviceDescription::FromProto(device_description_proto));
  device_description.set_dnn_version({1, 2, 3});
  AutotuneCacheKey key(device_description,
                       *module->entry_computation()->root_instruction());

  EXPECT_THAT(AutotunerCache::TryFindInCache(key, /*cache_dir=*/""),
              absl_testing::IsOkAndHolds(
                  Optional(EqualsProto(results.results(0).result()))))
      << "Cache key: " << key.ToString();
}

TEST_F(AutotunerCacheTest, LoadAutotuneResultsFromFile_TextProto2) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerCache::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerCache::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerCache::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());
}

TEST_F(AutotunerCacheTest, LoadAutotuneResultsFromFile_Protobuf) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));
  AutotunerCache::ClearAutotuneResults();
  EXPECT_TRUE(AutotunerCache::ResultCacheIsEmpty());

  TF_EXPECT_OK(AutotunerCache::LoadAutotuneResultsFromFile(kFilePath));
  EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());
}

TEST_F(AutotunerCacheTest, ResultConflictsAreDetected) {
  TF_EXPECT_OK(PopulateResultCache());
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(AutotunerCache::SerializeAutotuneResultsToFile(kFilePath));
  EXPECT_THAT(AutotunerCache::LoadAutotuneResultsFromFile(kFilePath),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr("Duplicate autotuning result")));
}

class FileBasedCacheTest : public AutotunerCacheTest {
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
    CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), std::string(filepath),
                                   &file_content));
    return file_content;
  }

  void Write(const absl::string_view filepath,
             const absl::string_view content) {
    CHECK_OK(CreateDirIfNeeded(cache_dir_, tsl::Env::Default()));
    CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), std::string(filepath),
                                    content));
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

  AutotuneCacheKey GetCacheKey() const {
    return AutotuneCacheKey(executor_->GetDeviceDescription(), *dot_);
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

TEST_F(FileBasedCacheTest, ResultsAreWrittenToAndReadFromFileCache) {
  AutotuneCacheKey key = GetCacheKey();
  const std::string cache_file_path = GetCacheFilePath();

  // At first, key is not in cache.
  EXPECT_THAT(GetFilesInDir(cache_dir_), IsEmpty());
  TF_ASSERT_OK_AND_ASSIGN(std::optional<AutotuneResult> result,
                          AutotunerCache::TryFindInCache(key, cache_dir_));
  EXPECT_FALSE(result.has_value());
  EXPECT_THAT(GetFilesInDir(cache_dir_), IsEmpty());

  // Add key to cache
  TF_ASSERT_OK_AND_ASSIGN(
      const AutotunerCache::ResultAndInserted result_and_inserted,
      AutotunerCache::AddResultToCaches(key, result1_, cache_dir_,
                                      GetCacheMode()));
  EXPECT_THAT(result_and_inserted.result, EqualsProto(result1_));
  EXPECT_TRUE(result_and_inserted.inserted);

  // Check that key is in file cache.
  EXPECT_THAT(Read(cache_file_path), HasSubstr(ToString(result1_)));
  EXPECT_THAT(GetFilesInDir(cache_dir_),
              UnorderedElementsAre(GetCacheFilename(), "tmp"));

  // Clear in-memory cache.
  AutotunerCache::ClearAutotuneResults();

  // Check that key is in on-disk cache and loaded into in-memory cache.
  TF_ASSERT_OK_AND_ASSIGN(result,
                          AutotunerCache::TryFindInCache(key, cache_dir_));
  EXPECT_TRUE(result.has_value());
  EXPECT_THAT(result.value(), EqualsProto(result1_));
}

TEST_F(FileBasedCacheTest, ResultsAreNotWrittenIfCacheModeIsRead) {
  SetCacheMode(DebugOptions::AUTOTUNE_CACHE_MODE_READ);
  AutotuneCacheKey key = GetCacheKey();

  // Add key to cache
  TF_ASSERT_OK_AND_ASSIGN(
      const AutotunerCache::ResultAndInserted result_and_inserted,
      AutotunerCache::AddResultToCaches(key, result1_, cache_dir_,
                                      GetCacheMode()));
  EXPECT_THAT(result_and_inserted.result, EqualsProto(result1_));
  EXPECT_TRUE(result_and_inserted.inserted);

  // Check that key is not in file cache, because cache mode is read-only.
  EXPECT_THAT(GetFilesInDir(cache_dir_), IsEmpty());
}

TEST_F(FileBasedCacheTest,
       AddResultToCachesDoesNotWriteToCacheDirIfItIsEmpty) {
  AutotuneCacheKey key = GetCacheKey();

  // Add key to cache
  TF_ASSERT_OK_AND_ASSIGN(
      const AutotunerCache::ResultAndInserted result_and_inserted,
      AutotunerCache::AddResultToCaches(key, result1_, /*cache_dir=*/"",
                                      GetCacheMode()));
  EXPECT_THAT(result_and_inserted.result, EqualsProto(result1_));
  EXPECT_TRUE(result_and_inserted.inserted);

  // Check that key is not in file cache, because cache_dir is empty.
  EXPECT_THAT(GetFilesInDir(cache_dir_), IsEmpty());
}

TEST_F(FileBasedCacheTest, AddResultToCachesDoesNotWriteTheSameKeyTwice) {
  AutotuneCacheKey key = GetCacheKey();
  const std::string cache_file_path = GetCacheFilePath();

  // Add key to cache
  TF_ASSERT_OK_AND_ASSIGN(
      AutotunerCache::ResultAndInserted result_and_inserted,
      AutotunerCache::AddResultToCaches(key, result1_, cache_dir_,
                                      GetCacheMode()));
  EXPECT_THAT(result_and_inserted.result, EqualsProto(result1_));
  EXPECT_TRUE(result_and_inserted.inserted);
  EXPECT_THAT(Read(cache_file_path), HasSubstr(ToString(result1_)));

  // Modify the cache file with result2_.
  Write(cache_file_path, ToString(result2_));
  EXPECT_THAT(Read(cache_file_path), HasSubstr(ToString(result2_)));

  // Try to add key to cache again with result1_.
  TF_ASSERT_OK_AND_ASSIGN(
      result_and_inserted,
      AutotunerCache::AddResultToCaches(key, result1_, cache_dir_,
                                      GetCacheMode()));
  EXPECT_THAT(result_and_inserted.result, EqualsProto(result1_));
  EXPECT_FALSE(result_and_inserted.inserted);

  // Check that cache file is not modified, because result_and_inserted.inserted
  // is false.
  EXPECT_THAT(Read(cache_file_path), HasSubstr(ToString(result2_)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
