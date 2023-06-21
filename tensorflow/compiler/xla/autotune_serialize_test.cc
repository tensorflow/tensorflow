/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/autotune_serialize.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::absl::LogSeverity;
using ::absl::ScopedMockLog;
using ::testing::EndsWith;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::StartsWith;
using ::testing::TempDir;

class AutotuneSerializeTest : public HloTestBase {
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
};

TEST_F(AutotuneSerializeTest, SerializeAutotuneResultsToFile_TextProto1) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotuneSerializeTest, SerializeAutotuneResultsToFile_TextProto2) {
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotuneSerializeTest, SerializeAutotuneResultsToFile_Protobuf) {
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(results.ParseFromString(autotune_results_str));
}

TEST_F(AutotuneSerializeTest, LoadAutotuneResultsFromFile_TextProto1) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(LoadAutotuneResultsFromFile(kFilePath));
}

TEST_F(AutotuneSerializeTest, LoadAutotuneResultsFromFile_TextProto2) {
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(LoadAutotuneResultsFromFile(kFilePath));
}

TEST_F(AutotuneSerializeTest, LoadAutotuneResultsFromFile_Protobuf) {
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(LoadAutotuneResultsFromFile(kFilePath));
}

TEST_F(AutotuneSerializeTest, LoadAutotuneResultsFromFileOnce) {
  // ScopedMockLog cannot catch the logs in the open source version.
  if (tsl::testing::kIsOpenSource) {
    return;
  }

  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(SerializeAutotuneResultsToFile(kFilePath));

  ScopedMockLog log;
  EXPECT_CALL(log, Log(LogSeverity::kInfo, EndsWith("/autotune_serialize.cc"),
                       StartsWith("Autotune results loaded from file:")))
      .Times(1);
  log.StartCapturingLogs();

  TF_EXPECT_OK(LoadAutotuneResultsFromFileOnce(kFilePath));
  TF_EXPECT_OK(LoadAutotuneResultsFromFileOnce(kFilePath));
}

}  // namespace
}  // namespace xla
