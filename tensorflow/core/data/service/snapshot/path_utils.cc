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

#include <cstdint>
#include <string>
#include <tuple>
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
constexpr const char kWorkerFileName[] = "owner_worker";
constexpr const char kSnapshotMetadataFileName[] = "snapshot.metadata";
constexpr const char kDatasetDefFileName[] = "dataset_def.proto";
constexpr const char kDatasetSpecFileName[] = "dataset_spec.pb";
constexpr const char kStreamsDirectoryName[] = "streams";
constexpr const char kSplitsDirectoryName[] = "splits";
constexpr const char kCheckpointsDirectoryName[] = "checkpoints";
constexpr const char kCommittedChunksDirectoryName[] = "chunks";
constexpr const char kUncommittedChunksDirectoryName[] = "uncommitted_chunks";
constexpr int64_t kUnknownNumElements = -1;

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

std::string RepetitionDirectory(absl::string_view snapshot_path,
                                int64_t stream_index, int64_t source_id,
                                int64_t repetition_index) {
  return tsl::io::JoinPath(
      SourceDirectory(snapshot_path, stream_index, source_id),
      absl::StrCat("repetition_", repetition_index));
}

std::string SplitPath(absl::string_view snapshot_path, int64_t stream_index,
                      int64_t source_id, int64_t repetition_index,
                      int64_t local_index, int64_t global_index) {
  return tsl::io::JoinPath(
      RepetitionDirectory(snapshot_path, stream_index, source_id,
                          repetition_index),
      absl::StrCat("split_", local_index, "_", global_index));
}

tsl::StatusOr<int64_t> ParseStreamDirectoryName(
    absl::string_view stream_directory_name) {
  std::vector<std::string> tokens = absl::StrSplit(stream_directory_name, '_');
  int64_t stream_index = 0;
  if (tokens.size() != 2 || tokens[0] != "stream" ||
      !absl::SimpleAtoi(tokens[1], &stream_index) || stream_index < 0) {
    return tsl::errors::InvalidArgument(
        "Invalid stream directory name: ", stream_directory_name,
        ". Expected stream_<stream_index>.");
  }
  return stream_index;
}

tsl::StatusOr<std::pair<int64_t, int64_t>> ParseSplitFilename(
    absl::string_view split_filename) {
  std::vector<std::string> tokens =
      absl::StrSplit(tsl::io::Basename(split_filename), '_');
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

tsl::StatusOr<std::pair<int64_t, int64_t>> ParseCheckpointFilename(
    absl::string_view checkpoint_filename) {
  std::vector<std::string> tokens = absl::StrSplit(checkpoint_filename, '_');
  int64_t checkpoint_index = 0, checkpoint_num_elements = 0;
  if (tokens.size() != 3 || tokens[0] != "checkpoint" ||
      !absl::SimpleAtoi(tokens[1], &checkpoint_index) || checkpoint_index < 0 ||
      !absl::SimpleAtoi(tokens[2], &checkpoint_num_elements) ||
      (checkpoint_num_elements < 0 &&
       checkpoint_num_elements != kUnknownNumElements)) {
    return tsl::errors::InvalidArgument(
        "Invalid checkpoint file name: ", checkpoint_filename,
        ". Expected checkpoint_<checkpoint_index>_<checkpoint_num_elements>.");
  }
  return std::make_pair(checkpoint_index, checkpoint_num_elements);
}

tsl::StatusOr<std::tuple<int64_t, int64_t, int64_t>> ParseChunkFilename(
    absl::string_view chunk_filename) {
  std::vector<std::string> tokens = absl::StrSplit(chunk_filename, '_');
  int64_t stream_index = 0, stream_chunk_index = 0, chunk_num_elements = 0;
  if (tokens.size() != 4 || tokens[0] != "chunk" ||
      !absl::SimpleAtoi(tokens[1], &stream_index) || stream_index < 0 ||
      !absl::SimpleAtoi(tokens[2], &stream_chunk_index) ||
      stream_chunk_index < 0 ||
      !absl::SimpleAtoi(tokens[3], &chunk_num_elements) ||
      (chunk_num_elements < 0 && chunk_num_elements != kUnknownNumElements)) {
    return tsl::errors::InvalidArgument(
        "Invalid chunk file name: ", chunk_filename,
        ". Expected "
        "chunk_<stream_index>_<stream_chunk_index>_<chunk_num_elements>.");
  }
  return std::make_tuple(stream_index, stream_chunk_index, chunk_num_elements);
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

std::string StreamWorkerFilePath(absl::string_view snapshot_path,
                                 int64_t stream_index) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_index),
                           kWorkerFileName);
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
