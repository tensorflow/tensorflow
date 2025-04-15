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
#include "tensorflow/core/data/service/snapshot/path_utils.h"

#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ::testing::Pair;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(PathUtilsTest, StreamsDirectory) {
  EXPECT_THAT(StreamsDirectory("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.streams"));
}

TEST(PathUtilsTest, StreamDirectory) {
  EXPECT_THAT(StreamDirectory("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0"));
}

TEST(PathUtilsTest, SplitsDirectory) {
  EXPECT_THAT(SplitsDirectory("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0.splits"));
}

TEST(PathUtilsTest, SourceDirectory) {
  EXPECT_THAT(
      SourceDirectory("/path/to/snapshot", /*stream_index=*/0,
                      /*source_index=*/1),
      MatchesRegex("/path/to/snapshot.streams.stream_0.splits.source_1"));
}

TEST(PathUtilsTest, RepetitionDirectory) {
  EXPECT_THAT(
      RepetitionDirectory("/path/to/snapshot", /*stream_index=*/0,
                          /*source_index=*/1, /*repetition_index=*/2),
      MatchesRegex(
          "/path/to/snapshot.streams.stream_0.splits.source_1.repetition_2"));
}

TEST(PathUtilsTest, SplitPath) {
  EXPECT_THAT(
      SplitPath("/path/to/snapshot", /*stream_index=*/0, /*source_index=*/1,
                /*repetition_index=*/2, /*local_index=*/3, /*global_index=*/4),
      MatchesRegex(
          "/path/to/"
          "snapshot.streams.stream_0.splits.source_1.repetition_2.split_3_4"));
}

TEST(PathUtilsTest, ParseStreamDirectoryName) {
  EXPECT_THAT(ParseStreamDirectoryName("stream_1"), IsOkAndHolds(1));
}

TEST(PathUtilsTest, ParseSourceDirectoryName) {
  EXPECT_THAT(ParseSourceDirectoryName("source_1"), IsOkAndHolds(1));
  EXPECT_THAT(ParseSourceDirectoryName(""),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected source_<source_index>")));
  EXPECT_THAT(ParseSourceDirectoryName("source_-1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected source_<source_index>")));
  EXPECT_THAT(ParseSourceDirectoryName("chunk_1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected source_<source_index>")));
}

TEST(PathUtilsTest, ParseRepetitionDirectoryName) {
  EXPECT_THAT(ParseRepetitionDirectoryName("repetition_1"), IsOkAndHolds(1));
  EXPECT_THAT(ParseRepetitionDirectoryName(""),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected repetition_<repetition_index>")));
  EXPECT_THAT(ParseRepetitionDirectoryName("repetition_-1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected repetition_<repetition_index>")));
  EXPECT_THAT(ParseRepetitionDirectoryName("chunk_1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected repetition_<repetition_index>")));
}

TEST(PathUtilsTest, InvalidStreamDirectoryName) {
  EXPECT_THAT(ParseStreamDirectoryName(""),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected stream_<stream_index>")));
  EXPECT_THAT(ParseStreamDirectoryName("stream_-1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected stream_<stream_index>")));
  EXPECT_THAT(ParseStreamDirectoryName("chunk_1"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected stream_<stream_index>")));
}

TEST(PathUtilsTest, ParseSplitFilename) {
  EXPECT_THAT(ParseSplitFilename("split_0_1"), IsOkAndHolds(Pair(0, 1)));
}

TEST(PathUtilsTest, InvalidSplitFilename) {
  EXPECT_THAT(
      ParseSplitFilename(""),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected split_<local_split_index>_<global_split_index>")));
  EXPECT_THAT(
      ParseSplitFilename("split_123"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected split_<local_split_index>_<global_split_index>")));
  EXPECT_THAT(
      ParseSplitFilename("split_-1_(-1)"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected split_<local_split_index>_<global_split_index>")));
  EXPECT_THAT(
      ParseSplitFilename("chunk_1_2"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected split_<local_split_index>_<global_split_index>")));
  EXPECT_THAT(
      ParseSplitFilename("split_5_0"),
      StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "The local split index 5 exceeds the global split index 0")));
}

TEST(PathUtilsTest, ParseCheckpointFilename) {
  EXPECT_THAT(ParseCheckpointFilename("checkpoint_0_1"),
              IsOkAndHolds(Pair(0, 1)));
  EXPECT_THAT(ParseCheckpointFilename("checkpoint_0_-1"),
              IsOkAndHolds(Pair(0, -1)));
}

TEST(PathUtilsTest, InvalidCheckpointFilename) {
  EXPECT_THAT(
      ParseCheckpointFilename(""),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected "
                   "checkpoint_<checkpoint_index>_<checkpoint_num_elements>")));
  EXPECT_THAT(
      ParseCheckpointFilename("checkpoint_123"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected "
                   "checkpoint_<checkpoint_index>_<checkpoint_num_elements>")));
  EXPECT_THAT(
      ParseCheckpointFilename("checkpoint_-1_(-1)"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected "
                   "checkpoint_<checkpoint_index>_<checkpoint_num_elements>")));
  EXPECT_THAT(
      ParseCheckpointFilename("chunk_1_2"),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr(
                   "Expected "
                   "checkpoint_<checkpoint_index>_<checkpoint_num_elements>")));
}

TEST(PathUtilsTest, ParseChunkFilename) {
  EXPECT_THAT(ParseChunkFilename("chunk_0_1_2"),
              IsOkAndHolds(FieldsAre(0, 1, 2)));
  EXPECT_THAT(ParseChunkFilename("chunk_0_1_-1"),
              IsOkAndHolds(FieldsAre(0, 1, -1)));
}

TEST(PathUtilsTest, InvalidChunkFilename) {
  EXPECT_THAT(ParseChunkFilename(""),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected "
                                 "chunk_<stream_index>_<stream_chunk_index>_<"
                                 "chunk_num_elements>")));
  EXPECT_THAT(ParseChunkFilename("chunk_123_0"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected "
                                 "chunk_<stream_index>_<stream_chunk_index>_<"
                                 "chunk_num_elements>")));
  EXPECT_THAT(ParseChunkFilename("chunk_-1_(-1)_0"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected "
                                 "chunk_<stream_index>_<stream_chunk_index>_<"
                                 "chunk_num_elements>")));
  EXPECT_THAT(ParseChunkFilename("split_1_2_3"),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Expected "
                                 "chunk_<stream_index>_<stream_chunk_index>_<"
                                 "chunk_num_elements>")));
}

TEST(PathUtilsTest, StreamDoneFilePath) {
  EXPECT_THAT(StreamDoneFilePath("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0.DONE"));
}

TEST(PathUtilsTest, StreamWorkerFilePath) {
  EXPECT_THAT(StreamWorkerFilePath("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0.owner_worker"));
  EXPECT_THAT(StreamWorkerFilePath("/path/to/snapshot/streams/stream_0"),
              MatchesRegex("/path/to/snapshot.streams.stream_0.owner_worker"));
}

TEST(PathUtilsTest, SnapshotDoneFilePath) {
  EXPECT_THAT(SnapshotDoneFilePath("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.DONE"));
}

TEST(PathUtilsTest, SnapshotErrorFilePath) {
  EXPECT_THAT(SnapshotErrorFilePath("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.ERROR"));
}

TEST(PathUtilsTest, SnapshotMetadataFilePath) {
  EXPECT_THAT(SnapshotMetadataFilePath("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.snapshot.metadata"));
}

TEST(PathUtilsTest, DatasetDefFilePath) {
  EXPECT_THAT(DatasetDefFilePath("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.dataset_def.proto"));
}

TEST(PathUtilsTest, DatasetSpefFilePath) {
  EXPECT_THAT(DatasetSpecFilePath("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.dataset_spec.pb"));
}

TEST(PathUtilsTest, CheckpointsDirectory) {
  EXPECT_THAT(CheckpointsDirectory("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0.checkpoints"));
}

TEST(PathUtilsTest, CommittedChunksDirectory) {
  EXPECT_THAT(CommittedChunksDirectory("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.chunks"));
}

TEST(PathUtilsTest, UncommittedChunksDirectory) {
  EXPECT_THAT(
      UncommittedChunksDirectory("/path/to/snapshot", /*stream_index=*/0),
      MatchesRegex("/path/to/snapshot.streams.stream_0.uncommitted_chunks"));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
