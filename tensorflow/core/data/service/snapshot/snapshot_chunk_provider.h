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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_CHUNK_PROVIDER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_CHUNK_PROVIDER_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/env.h"

namespace tensorflow {
namespace data {

// Provides the next chunk to read. Blocks until the next chunk is unavailable,
// or all the chunks have been read. This class is thread-safe.
class SnapshotChunkProvider {
 public:
  SnapshotChunkProvider(absl::string_view snapshot_path, tsl::Env* env);
  virtual ~SnapshotChunkProvider() = default;
  SnapshotChunkProvider(const SnapshotChunkProvider&) = delete;
  SnapshotChunkProvider& operator=(const SnapshotChunkProvider&) = delete;

  // Returns the absolute file path of next snapshot chunk to read. If there is
  // no available chunk, blocks until the next chunk is unavailable, or all the
  // chunks are read. Returns std::nullopt if all chunks have been read.
  absl::StatusOr<std::optional<std::string>> GetNext();

  // TODO(b/297930782): Support save/load.
  // TODO(b/297930782): Support cancellation.

 private:
  // State of the snapshot.
  struct SnapshotState {
    SnapshotState() = default;
    explicit SnapshotState(bool snapshot_is_done)
        : snapshot_is_done(snapshot_is_done) {}
    explicit SnapshotState(absl::Status status) : status(std::move(status)) {}

    // True if the snapshot is done without errors.
    bool snapshot_is_done = false;

    // Non-OK status if writing the snapshot fails.
    absl::Status status = absl::OkStatus();
  };

  // Updates the snapshot state and available chunks.
  absl::Status UpdateSnapshot();

  // Reads the DONE or ERROR file and returns a SnapshotState indicating whether
  // the snapshot is complete.
  absl::StatusOr<SnapshotState> GetSnapshotState();

  // Reads the available chunks from disk and returns a vector of chunk file
  // names.
  absl::StatusOr<std::vector<std::string>> GetAvailableChunks();

  const std::string snapshot_path_;
  tsl::Env* const env_;

  mutable absl::Mutex mu_;

  // The set of read chunks.
  absl::flat_hash_set<std::string> chunks_read_ ABSL_GUARDED_BY(mu_);

  // The set of unread chunks.
  absl::flat_hash_set<std::string> chunks_unread_ ABSL_GUARDED_BY(mu_);

  // State of the snapshot.
  SnapshotState snapshot_state_ ABSL_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_CHUNK_PROVIDER_H_
