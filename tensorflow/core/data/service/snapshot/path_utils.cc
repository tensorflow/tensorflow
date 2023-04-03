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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kDoneFileName[] = "DONE";
constexpr const char kErrorFileName[] = "ERROR";
constexpr const char kSnapshotMetadataFileName[] = "snapshot.metadata";
constexpr const char kDatasetDefFileName[] = "dataset_def.proto";
constexpr const char kDatasetSpecFileName[] = "dataset_spec.pb";
constexpr const char kStreamsDirectoryName[] = "streams";
constexpr const char kSplitsDirectoryName[] = "splits";
constexpr const char kCheckpointsDirectoryName[] = "checkpoints";
constexpr const char kCommittedChunksDirectoryName[] = "chunks";
constexpr const char kUncommittedChunksDirectoryName[] = "uncommitted_chunks";

}  // namespace

std::string StreamsDirectory(absl::string_view snapshot_path) {
  return tsl::io::JoinPath(snapshot_path, kStreamsDirectoryName);
}

std::string StreamDirectory(absl::string_view snapshot_path,
                            int64_t stream_index) {
  return tsl::io::JoinPath(StreamsDirectory(snapshot_path),
                           absl::StrCat("stream_", stream_index));
}

std::string SplitsDirectory(absl::string_view snapshot_path,
                            int64_t stream_index) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_index),
                           kSplitsDirectoryName);
}

std::string SourceDirectory(absl::string_view snapshot_path,
                            int64_t stream_index, int64_t source_id) {
  return tsl::io::JoinPath(SplitsDirectory(snapshot_path, stream_index),
                           absl::StrCat("source_", source_id));
}

std::string SplitPath(absl::string_view snapshot_path, int64_t stream_index,
                      int64_t source_id, int64_t local_index,
                      int64_t global_index) {
  return tsl::io::JoinPath(
      SourceDirectory(snapshot_path, stream_index, source_id),
      absl::StrCat("split_", local_index, "_", global_index));
}

tsl::StatusOr<std::pair<int64_t, int64_t>> SplitIndices(
    absl::string_view split_filename) {
  std::vector<std::string> tokens = absl::StrSplit(split_filename, '_');
  int64_t local_split_index = 0, global_split_index = 0;
  if (tokens.size() != 3 || tokens[0] != "split" ||
      !absl::SimpleAtoi(tokens[1], &local_split_index) ||
      local_split_index < 0 ||
      !absl::SimpleAtoi(tokens[2], &global_split_index) ||
      global_split_index < 0) {
    return tsl::errors::InvalidArgument(
        "Invalid split file name: ", split_filename,
        ". Expected split_<local_split_index>_<global_split_index>.");
  }
  if (local_split_index > global_split_index) {
    return tsl::errors::InvalidArgument(
        "Invalid split file name: ", split_filename, ". The local split index ",
        local_split_index, " exceeds the global split index ",
        global_split_index, ".");
  }
  return std::make_pair(local_split_index, global_split_index);
}

tsl::StatusOr<std::pair<int64_t, int64_t>> ChunkIndices(
    absl::string_view chunk_filename) {
  std::vector<std::string> tokens = absl::StrSplit(chunk_filename, '_');
  int64_t stream_index = 0, stream_chunk_index = 0;
  if (tokens.size() != 3 || tokens[0] != "chunk" ||
      !absl::SimpleAtoi(tokens[1], &stream_index) || stream_index < 0 ||
      !absl::SimpleAtoi(tokens[2], &stream_chunk_index) ||
      stream_chunk_index < 0) {
    return tsl::errors::InvalidArgument(
        "Invalid chunk file name: ", chunk_filename,
        ". Expected chunk_<stream_index>_<stream_chunk_index>.");
  }
  return std::make_pair(stream_index, stream_chunk_index);
}

std::string SnapshotMetadataFilePath(absl::string_view snapshot_path_) {
  return tsl::io::JoinPath(snapshot_path_, kSnapshotMetadataFileName);
}

std::string DatasetDefFilePath(absl::string_view snapshot_path_) {
  return tsl::io::JoinPath(snapshot_path_, kDatasetDefFileName);
}

std::string DatasetSpecFilePath(absl::string_view snapshot_path_) {
  return tsl::io::JoinPath(snapshot_path_, kDatasetSpecFileName);
}

std::string StreamDoneFilePath(absl::string_view snapshot_path,
                               int64_t stream_index) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_index),
                           kDoneFileName);
}

std::string SnapshotDoneFilePath(absl::string_view snapshot_path) {
  return tsl::io::JoinPath(snapshot_path, kDoneFileName);
}

std::string SnapshotErrorFilePath(absl::string_view snapshot_path) {
  return tsl::io::JoinPath(snapshot_path, kErrorFileName);
}

std::string CheckpointsDirectory(absl::string_view snapshot_path,
                                 int64_t stream_index) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_index),
                           kCheckpointsDirectoryName);
}

std::string CommittedChunksDirectory(absl::string_view snapshot_path) {
  return tsl::io::JoinPath(snapshot_path, kCommittedChunksDirectoryName);
}

std::string UncommittedChunksDirectory(absl::string_view snapshot_path,
                                       int64_t stream_index) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_index),
                           kUncommittedChunksDirectoryName);
}
}  // namespace data
}  // namespace tensorflow
