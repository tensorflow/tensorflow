/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/file_utils.h"

#include <string>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/status_to_from_proto.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

tsl::StatusOr<std::string> CreateTestDirectory() {
  std::string directory;
  if (!tsl::Env::Default()->LocalTempFilename(&directory)) {
    return tsl::errors::FailedPrecondition(
        "Failed to create local test directory.");
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(directory));
  return directory;
}

using AtomicallyWriteStringToFileTest = ::testing::TestWithParam<std::string>;

TEST_P(AtomicallyWriteStringToFileTest, WriteString) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  std::string file_contents = GetParam();
  TF_ASSERT_OK(AtomicallyWriteStringToFile(test_file, file_contents,
                                           tsl::Env::Default()));

  std::string data;
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(test_file));
  TF_ASSERT_OK(ReadFileToString(tsl::Env::Default(), test_file, &data));
  EXPECT_EQ(data, file_contents);
}

INSTANTIATE_TEST_SUITE_P(FileContents, AtomicallyWriteStringToFileTest,
                         ::testing::ValuesIn<std::string>({"OK", ""}));

TEST(FileUtilsTest, AtomicallyWriteBinaryProto) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  DatasetDef out = testing::RangeDataset(/*range=*/10);
  TF_ASSERT_OK(AtomicallyWriteBinaryProto(test_file, out, tsl::Env::Default()));

  DatasetDef in;
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(test_file));
  TF_ASSERT_OK(ReadBinaryProto(tsl::Env::Default(), test_file, &in));
  EXPECT_THAT(in, testing::EqualsProto(out));
}

TEST(FileUtilsTest, AtomicallyWriteTextProto) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  DatasetDef out = testing::RangeDataset(/*range=*/10);
  TF_ASSERT_OK(AtomicallyWriteTextProto(test_file, out, tsl::Env::Default()));

  DatasetDef in;
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(test_file));
  TF_ASSERT_OK(ReadTextProto(tsl::Env::Default(), test_file, &in));
  EXPECT_THAT(in, testing::EqualsProto(out));
}

TEST(FileUtilsTest, AtomicallyWriteTFRecord) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  Tensor out = CreateTensor<int64_t>(TensorShape({2}), {1, 2});
  TF_ASSERT_OK(AtomicallyWriteTFRecords(
      test_file, {out}, tsl::io::compression::kSnappy, tsl::Env::Default()));

  TF_EXPECT_OK(tsl::Env::Default()->FileExists(test_file));
  snapshot_util::TFRecordReaderImpl reader(test_file,
                                           tsl::io::compression::kSnappy);
  TF_ASSERT_OK(reader.Initialize(tsl::Env::Default()));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> in, reader.GetTensors());
  EXPECT_EQ(out.DebugString(), in.front().DebugString());
}

TEST(FileUtilsTest, GetChildren) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  TF_ASSERT_OK(AtomicallyWriteStringToFile(test_file, "", tsl::Env::Default()));
  std::string tmp_file = tsl::io::JoinPath(directory, "test_file.tmp");
  TF_ASSERT_OK(AtomicallyWriteStringToFile(tmp_file, "", tsl::Env::Default()));
  EXPECT_THAT(GetChildren(directory, tsl::Env::Default()),
              IsOkAndHolds(ElementsAre("test_file")));
}

TEST(FileUtilsTest, GetChildrenEmptyDirectory) {
  TF_ASSERT_OK_AND_ASSIGN(std::string empty_directory, CreateTestDirectory());
  EXPECT_THAT(GetChildren(empty_directory, tsl::Env::Default()),
              IsOkAndHolds(IsEmpty()));
}

TEST(FileUtilsTest, GetChildrenDirectoryNotFound) {
  EXPECT_THAT(GetChildren("Not exist", tsl::Env::Default()),
              StatusIs(tsl::error::NOT_FOUND));
}

TEST(FileUtilsTest, ValidateSnapshotSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string done_file = tsl::io::JoinPath(directory, "DONE");
  TF_ASSERT_OK(AtomicallyWriteStringToFile(done_file, "", tsl::Env::Default()));
  TF_EXPECT_OK(ValidateSnapshot(directory, tsl::Env::Default()));
}

TEST(FileUtilsTest, ValidateSnapshotError) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string error_file = tsl::io::JoinPath(directory, "ERROR");
  TF_ASSERT_OK(AtomicallyWriteTextProto(
      error_file,
      tsl::StatusToProto(
          errors::FailedPrecondition("Failed precondition test")),
      tsl::Env::Default()));
  EXPECT_THAT(ValidateSnapshot(directory, tsl::Env::Default()),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Failed precondition test")));
}

TEST(FileUtilsTest, ValidateSnapshotNotFinished) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  EXPECT_THAT(
      ValidateSnapshot(directory, tsl::Env::Default()),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("The save job has not finished writing the snapshot.")));
}

TEST(FileUtilsTest, ValidateSnapshotNotFound) {
  EXPECT_THAT(ValidateSnapshot("Not found", tsl::Env::Default()),
              StatusIs(tsl::error::NOT_FOUND));
}

TEST(FileUtilsTest, IsTemporaryFile) {
  EXPECT_TRUE(IsTemporaryFile("file.tmp"));
  EXPECT_FALSE(IsTemporaryFile("file"));
  EXPECT_FALSE(IsTemporaryFile(""));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
