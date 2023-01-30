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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/snapshot/test_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::ValuesIn;
using ::tsl::testing::IsOkAndHolds;

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

using SnapshotStreamWriterParameterizedTest =
    ::testing::TestWithParam<std::string>;

TEST_P(SnapshotStreamWriterParameterizedTest, SaveAndRestoreFromCheckpoints) {
  const int64_t range = 10;
  const std::string compression = GetParam();
  const DatasetDef dataset = testing::RangeDataset(range);
  const int64_t stream_index = 0;
  TF_ASSERT_OK_AND_ASSIGN(const std::string snapshot_path,
                          CreateSnapshotDirectory());
  TF_ASSERT_OK_AND_ASSIGN(
      testing::PartialSnapshotWriter partial_writer,
      testing::PartialSnapshotWriter::Create(dataset, snapshot_path,
                                             stream_index, compression));

  // Generates a snapshot directory with a single checkpoint. The stream writer
  // should resume from this checkpoint.
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));

  SnapshotWriterParams writer_params{snapshot_path,
                                     /*stream_index=*/0, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              IsOkAndHolds(UnorderedElementsAre(6, 7, 8, 9)));
}

INSTANTIATE_TEST_SUITE_P(Compression, SnapshotStreamWriterParameterizedTest,
                         ValuesIn<std::string>({tsl::io::compression::kNone,
                                                tsl::io::compression::kGzip,
                                                tsl::io::compression::kSnappy,
                                                tsl::io::compression::kZlib}));

TEST(SnapshotStreamWriterCheckpointTest, SyncCheckpointsWithChunksByRenaming) {
  const int64_t range = 10;
  const std::string compression = tsl::io::compression::kSnappy;
  const DatasetDef dataset = testing::RangeDataset(range);
  const int64_t stream_index = 0;
  TF_ASSERT_OK_AND_ASSIGN(const std::string snapshot_path,
                          CreateSnapshotDirectory());
  TF_ASSERT_OK_AND_ASSIGN(
      testing::PartialSnapshotWriter partial_writer,
      testing::PartialSnapshotWriter::Create(dataset, snapshot_path,
                                             stream_index, compression));

  // Unlike the previous test, this test has generated some uncommitted chunks.
  // The stream writer will first move the uncommitted chunks to the committed
  // chunks directory.
  TF_ASSERT_OK(partial_writer.WriteUncommittedChunks({0, 1, 2}));
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));

  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 6, 7, 8, 9)));
}

TEST(SnapshotStreamWriterCheckpointTest, SyncCheckpointsWithChunksByDeleting) {
  const int64_t range = 10;
  const std::string compression = tsl::io::compression::kNone;
  const DatasetDef dataset = testing::RangeDataset(range);
  const int64_t stream_index = 0;
  TF_ASSERT_OK_AND_ASSIGN(const std::string snapshot_path,
                          CreateSnapshotDirectory());
  // Generates some additional chunks.
  TF_ASSERT_OK_AND_ASSIGN(testing::PartialSnapshotWriter partial_writer,
                          testing::PartialSnapshotWriter::Create(
                              testing::RangeDataset(range + 5), snapshot_path,
                              stream_index, compression));

  // The writer will first delete uncommitted chunks after the checkpoint when
  // it is restored. In the end, only chunks 6--9 are written.
  TF_ASSERT_OK(
      partial_writer.WriteUncommittedChunks({6, 7, 8, 9, 10, 11, 12, 13, 14}));
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              IsOkAndHolds(UnorderedElementsAre(6, 7, 8, 9)));
}

TEST(SnapshotStreamWriterCheckpointTest, SyncCheckpointsWithChunks) {
  const int64_t range = 10;
  const std::string compression = tsl::io::compression::kZlib;
  const DatasetDef dataset = testing::RangeDataset(range);
  const int64_t stream_index = 0;
  TF_ASSERT_OK_AND_ASSIGN(const std::string snapshot_path,
                          CreateSnapshotDirectory());
  // Generates some additional chunks.
  TF_ASSERT_OK_AND_ASSIGN(testing::PartialSnapshotWriter partial_writer,
                          testing::PartialSnapshotWriter::Create(
                              testing::RangeDataset(range + 5), snapshot_path,
                              stream_index, compression));

  // This test combines the previous two test cases: Uncommitted chunks before
  // the checkpoint are committed; Uncommitted chunks after the checkpoint are
  // deleted.
  TF_ASSERT_OK(partial_writer.WriteUncommittedChunks(
      {1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}));
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              IsOkAndHolds(UnorderedElementsAre(1, 2, 5, 6, 7, 8, 9)));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
