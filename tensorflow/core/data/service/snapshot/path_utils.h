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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PATH_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PATH_UTILS_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace data {

// Returns the directory path for the assigned streams of a snapshot.
std::string StreamsDirectory(absl::string_view snapshot_path);

// Returns the directory path for a worker writing one stream of the snapshot.
std::string StreamDirectory(absl::string_view snapshot_path,
                            int64_t stream_index);

// Returns the directory path for the assigned splits for a worker writing one
// stream of a snapshot.
std::string SplitsDirectory(absl::string_view snapshot_path,
                            int64_t stream_index);

// Returns the directory path for the assigned splits for one source, for a
// worker writing one stream of a snapshot.
std::string SourceDirectory(absl::string_view snapshot_path,
                            int64_t stream_index, int64_t source_index);

// Returns the directory path for one repetition of a split provider.
std::string RepetitionDirectory(absl::string_view snapshot_path,
                                int64_t stream_index, int64_t source_index,
                                int64_t repetition_index);

// Returns the file path for an assigned split for a worker writing one stream
// of a snapshot.
std::string SplitPath(absl::string_view snapshot_path, int64_t stream_index,
                      int64_t source_index, int64_t repetition_index,
                      int64_t local_index, int64_t global_index);

// Returns the index of the stream. The expected format of
// `stream_directory_name` is:
// stream_<stream_index>
absl::StatusOr<int64_t> ParseStreamDirectoryName(
    absl::string_view stream_directory_name);

// Returns the index of the source. The expected format of
// `source_directory_name` is:
// source_<stream_index>
absl::StatusOr<int64_t> ParseSourceDirectoryName(
    absl::string_view source_directory_name);

// Returns the index of the repetition. The expected format of
// `repetition_directory_name` is:
// repetition_<stream_index>
absl::StatusOr<int64_t> ParseRepetitionDirectoryName(
    absl::string_view repetition_directory_name);

// Returns a pair of {local_split_index, global_split_index} of the split. The
// expected format of `split_filename` is:
// split_<local_split_index>_<global_split_index>
absl::StatusOr<std::pair<int64_t, int64_t>> ParseSplitFilename(
    absl::string_view split_filename);

// Returns a pair of {checkpoint_index, checkpoint_num_elements} of the
// checkpoint. The expected format of `checkpoint_filename` is:
// checkpoint_<checkpoint_index>_<checkpoint_num_elements>
absl::StatusOr<std::pair<int64_t, int64_t>> ParseCheckpointFilename(
    absl::string_view checkpoint_filename);

// Returns a tuple of {stream_index, stream_chunk_index, chunk_num_elements} of
// the chunk. The expected format of `chunk_filename` is:
// chunk_<stream_index>_<stream_chunk_index>_<chunk_num_elements>
absl::StatusOr<std::tuple<int64_t, int64_t, int64_t>> ParseChunkFilename(
    absl::string_view chunk_filename);

// Returns the path of the DONE file of a snapshot stream.
std::string StreamDoneFilePath(absl::string_view snapshot_path,
                               int64_t stream_index);

// Returns the path of the owner_worker file of a snapshot stream.
std::string StreamWorkerFilePath(absl::string_view snapshot_path,
                                 int64_t stream_index);

// Returns the path of the owner_worker file of a snapshot stream.
std::string StreamWorkerFilePath(absl::string_view stream_path);

// Returns the path of the DONE file of a snapshot.
std::string SnapshotDoneFilePath(absl::string_view snapshot_path);

// Returns the path of the ERROR file of a snapshot.
std::string SnapshotErrorFilePath(absl::string_view snapshot_path);

// Returns the path of the serialized metadata for a snapshot.
std::string SnapshotMetadataFilePath(absl::string_view snapshot_path);

// Returns the path of the serialized graph of the dataset for a snapshot.
std::string DatasetDefFilePath(absl::string_view snapshot_path);

// Returns the path of the serialized element spec of the dataset for a
// snapshot.
std::string DatasetSpecFilePath(absl::string_view snapshot_path);

// Returns the directory path for snapshot checkpoints.
std::string CheckpointsDirectory(absl::string_view snapshot_path,
                                 int64_t stream_index);

// Returns the directory path for committed chunks.
std::string CommittedChunksDirectory(absl::string_view snapshot_path);

// Returns the directory path for uncommitted chunks.
std::string UncommittedChunksDirectory(absl::string_view snapshot_path,
                                       int64_t stream_index);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PATH_UTILS_H_
