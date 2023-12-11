/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/snapshot_chunk_provider.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

SnapshotChunkProvider::SnapshotChunkProvider(absl::string_view snapshot_path,
                                             tsl::Env* env)
    : snapshot_path_(snapshot_path), env_(env) {}

absl::StatusOr<std::optional<std::string>> SnapshotChunkProvider::GetNext()
    ABSL_LOCKS_EXCLUDED(mu_) {
  while (true) {
    absl::MutexLock l(&mu_);
    if (!chunks_unread_.empty()) {
      std::string next_chunk = *chunks_unread_.begin();
      chunks_read_.insert(next_chunk);
      chunks_unread_.erase(next_chunk);
      return next_chunk;
    }
    if (snapshot_is_done_) {
      return std::nullopt;
    }
    TF_RETURN_IF_ERROR(UpdateChunks());
  }
}

absl::Status SnapshotChunkProvider::UpdateChunks()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // TODO(b/297930782): Handle failed snapshots.
  bool snapshot_is_done =
      env_->FileExists(SnapshotDoneFilePath(snapshot_path_)).ok();
  TF_ASSIGN_OR_RETURN(std::vector<std::string> chunks, GetAvailableChunks());

  // TODO(b/297930782): If no new chunks are updated, consider sleeping.
  for (absl::string_view chunk : chunks) {
    if (!chunks_read_.contains(chunk)) {
      chunks_unread_.insert(std::string(chunk));
    }
  }
  snapshot_is_done_ = snapshot_is_done;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>>
SnapshotChunkProvider::GetAvailableChunks() {
  absl::StatusOr<std::vector<std::string>> status_or_chunks =
      GetChildren(CommittedChunksDirectory(snapshot_path_), env_);
  if (status_or_chunks.ok()) {
    return *std::move(status_or_chunks);
  } else if (absl::IsNotFound(status_or_chunks.status())) {
    return std::vector<std::string>{};
  }
  return status_or_chunks.status();
}

}  // namespace data
}  // namespace tensorflow
