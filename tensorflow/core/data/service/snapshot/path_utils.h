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

#include <string>

namespace tensorflow {
namespace data {

// Returns the directory path for a worker writing one stream of the snapshot.
std::string StreamDirectory(const std::string& snapshot_path,
                            int64_t stream_id);

// Returns the directory path for snapshot checkpoints.
std::string CheckpointsDirectory(const std::string& snapshot_path,
                                 int64_t stream_id);

// Returns the directory path for committed chunks.
std::string CommittedChunksDirectory(const std::string& snapshot_path);

// Returns the directory path for uncommitted chunks.
std::string UncommittedChunksDirectory(const std::string& snapshot_path,
                                       int64_t stream_id);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PATH_UTILS_H_
