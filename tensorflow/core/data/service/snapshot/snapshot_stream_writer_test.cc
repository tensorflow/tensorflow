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
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class RangeIterator : public TaskIterator {
 public:
  explicit RangeIterator(const int64_t range) : range_(range) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    end_of_sequence = (next_ >= range_);
    if (end_of_sequence) {
      return OkStatus();
    }
    element = {Tensor{next_++}};
    return OkStatus();
  }

  int64_t Cardinality() const override { return range_; }

 private:
  const int64_t range_;
  int64_t next_ = 0;
};

StatusOr<std::string> CreateSnapshotDirectory() {
  std::string snapshot_path;
  if (!Env::Default()->LocalTempFilename(&snapshot_path)) {
    return errors::FailedPrecondition(
        "Failed to create local temp file for snapshot.");
  }
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(snapshot_path)));
  return snapshot_path;
}

StatusOr<std::unique_ptr<snapshot_util::Reader>> CreateSnapshotReader(
    const std::string& snapshot_path, int64_t num_elements,
    const std::string& compression, Env* env) {
  static constexpr int kTFRecordReader = 2;
  DataTypeVector dtypes(num_elements, DT_INT64);
  std::unique_ptr<snapshot_util::Reader> reader;
  TF_RETURN_IF_ERROR(snapshot_util::Reader::Create(
      env, snapshot_path, compression, kTFRecordReader, dtypes, &reader));
  return reader;
}

template <class T>
StatusOr<std::vector<T>> ReadSnapshot(const std::string& snapshot_path,
                                      const std::string& compression,
                                      int64_t num_elements) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<snapshot_util::Reader> reader,
                      CreateSnapshotReader(snapshot_path, num_elements,
                                           compression, Env::Default()));

  std::vector<Tensor> tensors;
  TF_RETURN_IF_ERROR(reader->ReadTensors(&tensors));

  std::vector<T> result;
  for (const Tensor& tensor : tensors) {
    result.push_back(tensor.unaligned_flat<T>().data()[0]);
  }
  return result;
}

using SnapshotStreamWriterParameterizedTest =
    ::testing::TestWithParam<std::string>;

TEST_P(SnapshotStreamWriterParameterizedTest, WriteSnapshot) {
  int64_t range = 10;
  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());

  SnapshotStreamWriter snapshot_writer(std::make_unique<RangeIterator>(range),
                                       snapshot_path, /*stream_id=*/0,
                                       compression, Env::Default());
  TF_ASSERT_OK(snapshot_writer.Wait());

  // The data is written to the committed chunks directory. The uncommitted
  // files are deleted.
  EXPECT_THAT(
      ReadSnapshot<int64_t>(
          tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path), "chunk_0"),
          compression, range),
      IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));

  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(UncommittedChunksDirectory(snapshot_path,
                                                               /*stream_id=*/0),
                                    "chunk_0"),
                  compression, range),
              StatusIs(error::NOT_FOUND));
}

TEST_P(SnapshotStreamWriterParameterizedTest, WriteSnapshotChunks) {
  int64_t range = 10;
  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());

  SnapshotStreamWriter snapshot_writer(
      std::make_unique<RangeIterator>(range), snapshot_path,
      /*stream_id=*/0, compression, Env::Default(),
      /*max_chunk_size_bytes=*/1);
  TF_ASSERT_OK(snapshot_writer.Wait());

  for (int i = 0; i < 10; ++i) {
    EXPECT_THAT(ReadSnapshot<int64_t>(
                    tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path),
                                      absl::StrCat("chunk_", i)),
                    compression,
                    /*num_elements=*/1),
                IsOkAndHolds(ElementsAre(i)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Compression, SnapshotStreamWriterParameterizedTest,
    testing::ValuesIn<std::string>({tsl::io::compression::kNone,
                                    tsl::io::compression::kGzip,
                                    tsl::io::compression::kSnappy,
                                    tsl::io::compression::kZlib}));

TEST(SnapshotStreamWriterTest, EmptyDataset) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotStreamWriter snapshot_writer(
      std::make_unique<RangeIterator>(0), snapshot_path, /*stream_id=*/0,
      tsl::io::compression::kSnappy, Env::Default());
  TF_ASSERT_OK(snapshot_writer.Wait());

  EXPECT_THAT(
      ReadSnapshot<int64_t>(
          tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path), "chunk_0"),
          tsl::io::compression::kSnappy, /*num_elements=*/0),
      IsOkAndHolds(IsEmpty()));
}

TEST(SnapshotStreamWriterTest, Cancel) {
  const int64_t range = 10000;
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());

  SnapshotStreamWriter snapshot_writer(
      std::make_unique<RangeIterator>(range), snapshot_path, /*stream_id=*/0,
      tsl::io::compression::kSnappy, Env::Default());
  snapshot_writer.Cancel();
  EXPECT_THAT(snapshot_writer.Wait(), StatusIs(error::CANCELLED));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
