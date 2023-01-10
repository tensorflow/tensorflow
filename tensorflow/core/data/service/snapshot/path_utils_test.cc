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

#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::MatchesRegex;

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
      SourceDirectory("/path/to/snapshot", /*stream_index=*/0, /*source_id=*/1),
      MatchesRegex("/path/to/snapshot.streams.stream_0.splits.source_1"));
}

TEST(PathUtilsTest, SplitPath) {
  EXPECT_THAT(
      SplitPath("/path/to/snapshot", /*stream_index=*/0, /*source_id=*/1,
                /*local_index=*/2, /*global_index=*/3),
      MatchesRegex(
          "/path/to/snapshot.streams.stream_0.splits.source_1.split_2_3"));
}

TEST(PathUtilsTest, CheckpointsDirectory) {
  EXPECT_THAT(CheckpointsDirectory("/path/to/snapshot", /*stream_index=*/0),
              MatchesRegex("/path/to/snapshot.streams.stream_0.checkpoints"));
}

TEST(PathUtilsTest, CommittedChunksDirectory) {
  EXPECT_THAT(CommittedChunksDirectory("/path/to/snapshot"),
              MatchesRegex("/path/to/snapshot.committed_chunks"));
}

TEST(PathUtilsTest, UncommittedChunksDirectory) {
  EXPECT_THAT(
      UncommittedChunksDirectory("/path/to/snapshot", /*stream_index=*/0),
      MatchesRegex("/path/to/snapshot.streams.stream_0.uncommitted_chunks"));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
