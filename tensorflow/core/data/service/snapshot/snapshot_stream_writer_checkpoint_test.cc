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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/compression.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/snapshot/test_utils.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tsl/platform/random.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;
using ::testing::ValuesIn;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

absl::StatusOr<std::string> CreateSnapshotDirectory() {
  std::string snapshot_path;
  if (!Env::Default()->LocalTempFilename(&snapshot_path)) {
    return absl::FailedPreconditionError(
        "Failed to create local temp file for snapshot.");
  }
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(snapshot_path)));
  return snapshot_path;
}

absl::StatusOr<int64_t> NumCheckpoints(const std::string& snapshot_path,
                                       int64_t stream_index) {
  std::string checkpoints_directory =
      CheckpointsDirectory(snapshot_path, stream_index);
  std::vector<std::string> checkpoint_filenames;
  if (!Env::Default()->FileExists(checkpoints_directory).ok()) {
    return 0;
  }
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(checkpoints_directory,
                                                 &checkpoint_filenames));
  return checkpoint_filenames.size();
}

using SnapshotStreamWriterParameterizedTest =
    ::testing::TestWithParam<std::string>;

TEST_P(SnapshotStreamWriterParameterizedTest, SaveAndRestoreFromCheckpoint) {
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
  TF_ASSERT_OK(partial_writer.WriteCommittedChunks({4}));

  SnapshotWriterParams writer_params{snapshot_path,
                                     /*stream_index=*/0, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshot_path, compression),
      absl_testing::IsOkAndHolds(UnorderedElementsAre(4, 5, 6, 7, 8, 9)));
}

TEST_P(SnapshotStreamWriterParameterizedTest,
       SaveAndRestoreFromEndOfSequenceCheckpoint) {
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

  // Each chunk contains 1 element. There are 10 chunks. The 10th checkpoint is
  // for the iterator that has reached the end of sequence.
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({10}));
  TF_ASSERT_OK(partial_writer.WriteCommittedChunks({9}));

  SnapshotWriterParams writer_params{snapshot_path,
                                     /*stream_index=*/0, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));

  // Since the end-of-sequence iterator is checkpointed, no more elements are
  // written here.
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(UnorderedElementsAre(9)));
}

TEST_P(SnapshotStreamWriterParameterizedTest, VaryingCheckpointInterval) {
  const int64_t range = 10;
  const std::string compression = GetParam();
  const DatasetDef dataset = testing::RangeDataset(range);
  const int64_t stream_index = 0;
  TF_ASSERT_OK_AND_ASSIGN(const std::string snapshot_path,
                          CreateSnapshotDirectory());
  TF_ASSERT_OK_AND_ASSIGN(
      testing::PartialSnapshotWriter partial_writer,
      testing::PartialSnapshotWriter::Create(
          dataset, snapshot_path, stream_index, compression,
          /*max_chunk_size=*/ByteSize::Bytes(1), /*checkpoint_interval=*/
          absl::Milliseconds(tsl::random::New64() % 1000)));
  TF_ASSERT_OK(
      partial_writer.WriteCommittedChunks({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  TF_ASSERT_OK(
      partial_writer.WriteCheckpoints({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

  SnapshotWriterParams writer_params{
      snapshot_path,
      /*stream_index=*/0,
      compression,
      Env::Default(),
      /*max_chunk_size=*/ByteSize::Bytes(1),
      /*checkpoint_interval=*/absl::Milliseconds(tsl::random::New64() % 1000),
      /*test_only_keep_temp_files=*/true};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

INSTANTIATE_TEST_SUITE_P(Compression, SnapshotStreamWriterParameterizedTest,
                         ValuesIn<std::string>({tsl::io::compression::kNone,
                                                tsl::io::compression::kGzip,
                                                tsl::io::compression::kSnappy,
                                                tsl::io::compression::kZlib}));

TEST(SnapshotStreamWriterCheckpointTest, SingleCheckpoint) {
  int64_t range = 10;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(testing::RangeDataset(range)));

  const std::string compression = tsl::io::compression::kSnappy;
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{snapshot_path,
                                     /*stream_index=*/0,
                                     compression,
                                     Env::Default(),
                                     /*max_chunk_size=*/ByteSize::Bytes(1),
                                     /*checkpoint_interval=*/absl::Hours(1),
                                     /*test_only_keep_temp_files=*/true};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
  EXPECT_THAT(NumCheckpoints(snapshot_path, /*stream_index=*/0),
              absl_testing::IsOkAndHolds(1));
}

TEST(SnapshotStreamWriterCheckpointTest, MultipleCheckpoints) {
  int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(testing::RangeDataset(range)));

  const std::string compression = tsl::io::compression::kSnappy;
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{
      snapshot_path,
      /*stream_index=*/0,
      compression,
      Env::Default(),
      /*max_chunk_size=*/ByteSize::Bytes(1),
      /*checkpoint_interval=*/absl::Microseconds(1),
      /*test_only_keep_temp_files=*/true};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4)));

  // Wrote one checkpoint for each element, and one for end_of_sequence.
  EXPECT_THAT(NumCheckpoints(snapshot_path, /*stream_index=*/0),
              absl_testing::IsOkAndHolds(range + 1));
}

TEST(SnapshotStreamWriterCheckpointTest, CleanupCheckpoint) {
  int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(testing::RangeDataset(range)));

  const std::string compression = tsl::io::compression::kSnappy;
  TF_ASSERT_OK_AND_ASSIGN(std::string snapshot_path, CreateSnapshotDirectory());
  SnapshotWriterParams writer_params{
      snapshot_path,
      /*stream_index=*/0,
      compression,
      Env::Default(),
      /*max_chunk_size=*/ByteSize::Bytes(20),
      /*checkpoint_interval=*/absl::Microseconds(1),
      /*test_only_keep_temp_files=*/false};
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  EXPECT_THAT(snapshot_writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4)));

  // If test_only_keep_temp_files is false (default), the checkpoints should be
  // removed once the snapshot is complete.
  EXPECT_THAT(NumCheckpoints(snapshot_path, /*stream_index=*/0),
              absl_testing::IsOkAndHolds(0));
}

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

  // This test has generated some uncommitted chunks. The stream writer will
  // first move the uncommitted chunks to the committed chunks directory.
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));
  TF_ASSERT_OK(partial_writer.WriteUncommittedChunks({0, 1, 2, 3, 4}));

  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotStreamWriterCheckpointTest,
     SyncCheckpointsWithSingleChunkByRenaming) {
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

  TF_ASSERT_OK(partial_writer.WriteUncommittedChunks({0}));
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({0}));

  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
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
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));
  TF_ASSERT_OK(partial_writer.WriteCommittedChunks({4}));
  TF_ASSERT_OK(
      partial_writer.WriteUncommittedChunks({6, 7, 8, 9, 10, 11, 12, 13, 14}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshot_path, compression),
      absl_testing::IsOkAndHolds(UnorderedElementsAre(4, 5, 6, 7, 8, 9)));
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
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));
  TF_ASSERT_OK(partial_writer.WriteCommittedChunks({0}));
  TF_ASSERT_OK(partial_writer.WriteUncommittedChunks(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path, compression),
              absl_testing::IsOkAndHolds(
                  UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotStreamWriterCheckpointTest, LostChunks) {
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

  // There is a checkpoint 5 but no chunks [2, 5). Should report lost chunks.
  TF_ASSERT_OK(partial_writer.WriteCheckpoints({5}));
  TF_ASSERT_OK(partial_writer.WriteCommittedChunks({0, 1}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StandaloneTaskIterator> iterator,
                          testing::TestIterator(dataset));
  SnapshotWriterParams writer_params{snapshot_path, stream_index, compression,
                                     Env::Default()};
  SnapshotStreamWriter writer(writer_params, std::move(iterator));
  EXPECT_THAT(writer.Wait(), absl_testing::StatusIs(
                                 absl::StatusCode::kInternal,
                                 HasSubstr("Unable to find chunks [2, 5).")));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
