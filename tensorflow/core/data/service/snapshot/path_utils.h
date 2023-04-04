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
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/statusor.h"

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
                            int64_t stream_index, int64_t source_id);

// Returns the file path for an assigned split for a worker writing one stream
// of a snapshot.
std::string SplitPath(absl::string_view snapshot_path, int64_t stream_index,
                      int64_t source_id, int64_t local_index,
                      int64_t global_index);

// Returns a pair of {local_split_index, global_split_index} of the split. The
// expected format of `split_filename` is:
// split_<local_split_index>_<global_split_index>
tsl::StatusOr<std::pair<int64_t, int64_t>> SplitIndices(
    absl::string_view split_filename);

// Returns a pair of {stream_index, stream_chunk_index} of the chunk. The
// expected format of `chunk_filename` is:
// chunk_<stream_index>_<stream_chunk_index>
tsl::StatusOr<std::pair<int64_t, int64_t>> ChunkIndices(
    absl::string_view chunk_filename);

// Returns the path of the DONE file of a snapshot stream.
std::string StreamDoneFilePath(absl::string_view snapshot_path,
                               int64_t stream_index);

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
