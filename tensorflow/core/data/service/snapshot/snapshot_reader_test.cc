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
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
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

template <class T>
StatusOr<std::vector<T>> ReadSnapshot(const std::string& base_path,
                                      const std::string& compression) {
  experimental::DistributedSnapshotMetadata metadata;
  metadata.set_compression(compression);
  SnapshotReaderParams params{base_path, metadata, DataTypeVector{DT_INT64},
                              Env::Default()};
  SnapshotReader reader(params);
  std::vector<T> result;
  while (true) {
    TF_ASSIGN_OR_RETURN(GetNextResult next, reader.GetNext());
    if (next.end_of_sequence) {
      return result;
    }
    result.push_back(next.tensors[0].unaligned_flat<T>().data()[0]);
  }
  return result;
}

TEST(SnapshotReaderTest, ReadSnapshot) {
  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(range)));
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     tsl::io::compression::kNone,
                                     Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotReaderTest, MultipleWritersAndChunks) {
  int64_t range = 10, num_writers = 3;
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());

  std::vector<std::unique_ptr<SnapshotStreamWriter>> writers;
  for (int i = 0; i < num_writers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                            TestIterator(testing::RangeDataset(range)));
    SnapshotWriterParams writer_params{
        snapshot_path, /*stream_index=*/i, tsl::io::compression::kNone,
        Env::Default(), /*max_chunk_size_bytes=*/1};
    writers.push_back(std::make_unique<SnapshotStreamWriter>(
        writer_params, std::move(iterator)));
  }
  for (std::unique_ptr<SnapshotStreamWriter>& writer : writers) {
    EXPECT_THAT(writer->Wait(), IsOkAndHolds(true));
  }
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
                                                1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
                                                2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotReaderTest, EmptyDataset) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          TestIterator(testing::RangeDataset(0)));
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path, /*stream_index=*/0,
                                     tsl::io::compression::kNone,
                                     Env::Default()};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));

  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(IsEmpty()));
}

TEST(SnapshotReaderTest, SnapshotDoesNotExist) {
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_does_not_exist,
                          CreateSnapshotDirectory());
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_does_not_exist,
                                    tsl::io::compression::kNone),
              StatusIs(error::NOT_FOUND));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
