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

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/env.h"

namespace tensorflow {
namespace data {

// Provides the next chunk to read. Blocks until the next chunk is unavailable,
// or all the chunks have been read. This class is thread-safe.
class SnapshotChunkProvider : public SplitProvider {
 public:
  SnapshotChunkProvider(absl::string_view snapshot_path, tsl::Env* env);
  ~SnapshotChunkProvider() override = default;
  SnapshotChunkProvider(const SnapshotChunkProvider&) = delete;
  SnapshotChunkProvider& operator=(const SnapshotChunkProvider&) = delete;

  // Returns the absolute file path of next snapshot chunk to read. If there is
  // no available chunk, blocks until the next chunk is unavailable, or all the
  // chunks are read. Sets `end_of_splits` to true if all chunks have been read.
  absl::Status GetNext(Tensor* split, bool* end_of_splits) override;

  absl::Status Reset() override;

  // Supports checkpointing.
  absl::Status Save(std::function<std::string(std::string)> full_name,
                    IteratorStateWriter* writer) override;
  absl::Status Restore(std::function<std::string(std::string)> full_name,
                       IteratorStateReader* reader) override;

  // If the snapshot is finished, returns the number of committed chunks.
  // If the snapshot is unfinished or has failed, returns kUnknownCardinality.
  int64_t Cardinality() const override;

  // Cancels the provider. After cancelling, if the snapshot is unfinished,
  // in-flight `GetNext` calls will return Cancelled status.
  void Cancel() override;

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

  // Used to sort chunks by chunk indexes so that chunks are read evenly across
  // streams and chunks of early repetitions are read first.
  struct ChunkOrder {
    bool operator()(const std::string& chunk1, const std::string& chunk2) const;
  };
  using OrderedChunkSet = absl::btree_set<std::string, ChunkOrder>;

  // String conversions to support `Save` and `Restore`.
  static std::string SetToString(const OrderedChunkSet& s);
  static OrderedChunkSet SetFromString(absl::string_view s);

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
  OrderedChunkSet chunks_read_ ABSL_GUARDED_BY(mu_);

  // The set of unread chunks. Uses an ordered set to make sure repeated reads
  // produce data in a deterministic order.
  OrderedChunkSet chunks_unread_ ABSL_GUARDED_BY(mu_);

  // State of the snapshot.
  SnapshotState snapshot_state_ ABSL_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_CHUNK_PROVIDER_H_
