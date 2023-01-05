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

#include "absl/strings/str_cat.h"
#include "tensorflow/tsl/platform/path.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kCheckpointsDirectoryName[] = "checkpoints";
constexpr const char kCommittedChunksDirectoryName[] = "committed_chunks";
constexpr const char kUncommittedChunksDirectoryName[] = "uncommitted_chunks";

}  // namespace

std::string StreamDirectory(const std::string& snapshot_path,
                            int64_t stream_id) {
  return tsl::io::JoinPath(snapshot_path, "streams",
                           absl::StrCat("stream_", stream_id));
}

std::string CheckpointsDirectory(const std::string& snapshot_path,
                                 int64_t stream_id) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_id),
                           kCheckpointsDirectoryName);
}

std::string CommittedChunksDirectory(const std::string& snapshot_path) {
  return tsl::io::JoinPath(snapshot_path, kCommittedChunksDirectoryName);
}

std::string UncommittedChunksDirectory(const std::string& snapshot_path,
                                       int64_t stream_id) {
  return tsl::io::JoinPath(StreamDirectory(snapshot_path, stream_id),
                           kUncommittedChunksDirectoryName);
}
}  // namespace data
}  // namespace tensorflow
