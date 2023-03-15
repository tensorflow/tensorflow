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
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/lib/monitoring/cell_reader.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::ValuesIn;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

StatusOr<std::unique_ptr<StandaloneTaskIterator>> TestIterator(
    const DatasetDef& dataset_def) {
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_RETURN_IF_ERROR(dataset->MakeIterator(&iterator));
  return std::make_unique<StandaloneTaskIterator>(std::move(dataset),
                                                  std::move(iterator));
}

template <class T>
class ElementOrErrorIterator : public TaskIterator {
 public:
  explicit ElementOrErrorIterator(const std::vector<StatusOr<T>>& elements)
      : elements_(elements) {}

  Status GetNext(std::vector<Tensor>& element, bool& end_of_sequence) override {
    end_of_sequence = (next_ >= elements_.size());
    if (end_of_sequence) {
      return OkStatus();
    }
    const StatusOr<T>& next_element = elements_[next_++];
    TF_RETURN_IF_ERROR(next_element.status());
    element = {Tensor{*next_element}};
    return OkStatus();
  }

  StatusOr<Tensor> Save() override { return Tensor(); }

  Status Restore(const Tensor& saved_iterator) override { return OkStatus(); }

  int64_t Cardinality() const override { return elements_.size(); }

 private:
  const std::vector<StatusOr<T>> elements_;
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

StatusOr<std::string> ReadStringFromFile(const std::string& filename) {
  std::string data;
  TF_RETURN_IF_ERROR(ReadFileToString(Env::Default(), filename, &data));
  return data;
}

using SnapshotStreamWriterParameterizedTest =
    ::testing::TestWithParam<std::string>;

TEST_P(SnapshotStreamWriterParameterizedTest, WriteSnapshot) {
  CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/snapshot_bytes_committed");
  EXPECT_EQ(cell_reader.Delta(), 0);

  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));

  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     compression, Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  // The data is written to the committed chunks directory. The uncommitted
  // files are deleted.
  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(writer_params.CommittedChunksDirectory(),
                                    "chunk_0_0"),
                  compression, range),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));

  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(writer_params.UncommittedChunksDirectory(),
                                    "chunk_0"),
                  compression, range),
              StatusIs(error::NOT_FOUND));
  // Writes at least 10 elements of 8 bytes.
  EXPECT_GE(cell_reader.Delta(), 80);
}

TEST_P(SnapshotStreamWriterParameterizedTest, StreamAlreadyCompleted) {
  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));

  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     compression, Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(writer_params.CommittedChunksDirectory(),
                                    "chunk_0_0"),
                  compression, range),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));

  // Writes the same snapshot.
  TF_ASSERT_OK_AND_ASSIGN(iterator, TestIterator(testing::RangeDataset(range)));
  SnapshotStreamWriter duplicate_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));
  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(writer_params.CommittedChunksDirectory(),
                                    "chunk_0_0"),
                  compression, range),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST_P(SnapshotStreamWriterParameterizedTest, WriteSnapshotChunks) {
  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));

  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     compression, Env::Default(),
                                     /*max_chunk_size_bytes=*/1};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  for (int i = 0; i < 10; ++i) {
    EXPECT_THAT(ReadSnapshot<int64_t>(
                    tsl::io::JoinPath(writer_params.CommittedChunksDirectory(),
                                      absl::StrCat("chunk_0_", i)),
                    compression,
                    /*num_elements=*/1),
                IsOkAndHolds(ElementsAre(i)));
  }
}

TEST_P(SnapshotStreamWriterParameterizedTest, WriteDoneFile) {
  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));

  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  std::string done_file_path = tsl::io::JoinPath(
      StreamDirectory(snapshot_path, /*stream_index=*/0), "DONE");
  std::string error_file_path = tsl::io::JoinPath(
      StreamDirectory(snapshot_path, /*stream_index=*/0), "ERROR");

  EXPECT_THAT(Env::Default()->FileExists(done_file_path),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(Env::Default()->FileExists(error_file_path),
              StatusIs(error::NOT_FOUND));
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     compression, Env::Default(),
                                     /*max_chunk_size_bytes=*/1};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));
  TF_EXPECT_OK(Env::Default()->FileExists(done_file_path));
  EXPECT_THAT(Env::Default()->FileExists(error_file_path),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(snapshot_writer.Completed(), IsOkAndHolds(true));
}

TEST_P(SnapshotStreamWriterParameterizedTest, WriteErrorFile) {
  auto error_iterator = std::make_unique<ElementOrErrorIterator<tstring>>(
      std::vector<StatusOr<tstring>>{
          tstring("First element"), errors::InvalidArgument("Invalid argument"),
          tstring("Second element"), errors::Aborted("Aborted")});
  std::string compression = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  std::string done_file_path = tsl::io::JoinPath(
      StreamDirectory(snapshot_path, /*stream_index=*/0), "DONE");
  std::string error_file_path = tsl::io::JoinPath(
      StreamDirectory(snapshot_path, /*stream_index=*/0), "ERROR");

  EXPECT_THAT(Env::Default()->FileExists(done_file_path),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(Env::Default()->FileExists(error_file_path),
              StatusIs(error::NOT_FOUND));
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     compression, Env::Default(),
                                     /*max_chunk_size_bytes=*/1};
  SnapshotStreamWriter snapshot_writer(writer_params,
                                       std::move(error_iterator));
  EXPECT_THAT(snapshot_writer.Wait(), StatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(Env::Default()->FileExists(done_file_path),
              StatusIs(error::NOT_FOUND));
  TF_EXPECT_OK(Env::Default()->FileExists(error_file_path));
  EXPECT_THAT(ReadStringFromFile(error_file_path),
              IsOkAndHolds(HasSubstr("Invalid argument")));
  EXPECT_THAT(snapshot_writer.Completed(), StatusIs(error::INVALID_ARGUMENT));
}

INSTANTIATE_TEST_SUITE_P(Compression, SnapshotStreamWriterParameterizedTest,
                         ValuesIn<std::string>({tsl::io::compression::kNone,
                                                tsl::io::compression::kGzip,
                                                tsl::io::compression::kSnappy,
                                                tsl::io::compression::kZlib}));

TEST(SnapshotStreamWriterTest, EmptyDataset) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(0)));

  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     tsl::io::compression::kSnappy,
                                     Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  EXPECT_THAT(ReadSnapshot<int64_t>(
                  tsl::io::JoinPath(writer_params.CommittedChunksDirectory(),
                                    "chunk_0_0"),
                  tsl::io::compression::kSnappy, /*num_elements=*/0),
              IsOkAndHolds(IsEmpty()));
}

TEST(SnapshotStreamWriterTest, Cancel) {
  const int64_t range = 10000;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));

  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     tsl::io::compression::kSnappy,
                                     Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  snapshot_writer.Cancel();
  EXPECT_THAT(snapshot_writer.Wait(), StatusIs(error::CANCELLED));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
