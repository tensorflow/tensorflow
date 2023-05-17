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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_MANAGER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_MANAGER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {

// A helper used by `DataServiceDispatcherImpl` to manage a call to `Snapshot`.
//
// Two mirrored states are maintained:
// - An in-memory state (objects in the `SnapshotManager` instance).
// - An on-disk state (files in the `SnapshotManager::path_`).
//
// The on-disk state has this structure:
// - snapshot_path
//   - DONE
//   - ERROR
//   - snapshot.metadata
//   - dataset_def.proto
//   - dataset_spec.pb
//   - chunks
//     - chunk_<stream_index>_<stream_chunk_index>_<num_elements>
//   - streams
//     - stream_0
//       - DONE
//       - ERROR
//       - splits
//         - source_0
//           - split_<local_split_index>_<global_split_index>
//       - uncommitted_chunks
//         - chunk_<chunk_index>
//       - checkpoints
//         - checkpoint_<chunk_index>_<num_elements>
//
class SnapshotManager {
 public:
  // Initiates a new snapshot process, creating a fresh in-memory state and
  // writing an on-disk state to `path`. Returns an error if `path` already
  // exists in the filesystem.
  static tsl::StatusOr<std::unique_ptr<SnapshotManager>> Start(
      const SnapshotRequest& request, Env* env);
  // Resumes an existing snapshot process, reading from the on-disk state in
  // `path` to derive an in-memory state. Returns an error if `path` is in a bad
  // state.
  static tsl::StatusOr<std::unique_ptr<SnapshotManager>> Resume(
      absl::string_view path, Env* env);

  // Handles the work pertaining to this snapshot process for the respective
  // `DispatcherService` API calls:
  // - `WorkerHeartbeat`: Returns a stream assignment for the worker.
  // - `GetSnapshotSplit`: Returns a split assignment for the worker.
  // - `GetSnapshotStreams`: Returns information about all streams.
  tsl::Status WorkerHeartbeat(const WorkerHeartbeatRequest& request,
                              WorkerHeartbeatResponse& response);
  tsl::Status GetSnapshotSplit(const GetSnapshotSplitRequest& request,
                               GetSnapshotSplitResponse& response);
  tsl::Status GetSnapshotStreams(GetSnapshotStreamsResponse& response);

 private:
  SnapshotManager(
      absl::string_view path, Env* env,
      std::optional<absl::Duration> resume_time_micros = std::nullopt)
      : path_(path), env_(env), resume_time_micros_(resume_time_micros) {}

  // Helpers for `Start` above. These update the on-disk state.
  tsl::Status Start(const SnapshotRequest& request);
  tsl::Status WriteOnDiskSkeleton();
  tsl::Status WriteOnDiskMetadata(const SnapshotRequest& request);

  // Helpers for `Resume` above. These update the in-memory state.
  tsl::Status Resume();
  tsl::Status ReadOnDiskMetadata();
  tsl::Status ReadOnDiskStreams();
  tsl::Status ReadOnDiskStream(
      int64_t stream_index, absl::flat_hash_set<int64_t>& global_split_indices);
  tsl::Status ReadOnDiskSource(
      int64_t stream_index, int64_t source_index,
      absl::flat_hash_set<int64_t>& global_split_indices);
  tsl::Status ReadOnDiskSplit(
      int64_t source_index, const std::vector<std::string>& split_files,
      const std::string& split_file,
      absl::flat_hash_set<int64_t>& global_split_indices);
  tsl::Status SkipSplit(SplitProvider& split_provider);

  // Helpers for `WorkerHeartbeat` above. These may update the in-memory and
  // on-disk states.
  tsl::StatusOr<std::optional<int64_t>> MaybeGetOrCreateStreamAssignment(
      absl::string_view worker_address,
      const SnapshotTaskProgress* snapshot_progress);
  tsl::Status HandleStreamCompletion(int64_t stream_index,
                                     absl::string_view worker_address);
  void ReassignPreviouslyAssignedStream(int64_t stream_index,
                                        absl::string_view worker_address);
  std::optional<int64_t> MaybeAssignOrphanStream(
      absl::string_view worker_address);
  tsl::StatusOr<int64_t> CreateAndAssignNewStream(
      absl::string_view worker_address);
  Status HandleStreamError(absl::string_view worker_address,
                           const StatusProto& status_proto);

  // The filepath of the on-disk state.
  const std::string path_;
  // A tensorflow environment interface used to write to and read from `path_`.
  tsl::Env* const env_;
  // Distributed snapshot metadata.
  experimental::DistributedSnapshotMetadata metadata_;
  // If `Resume`d, the timestamp of the resumption of the snapshot.
  std::optional<absl::Duration> resume_time_micros_;

  // The addresses of all workers considered to be dead based on heartbeat
  // timeout.
  absl::flat_hash_set<std::string> dead_workers_;

  struct Stream {
    explicit Stream(int64_t num_sources)
        : num_assigned_splits_per_source(num_sources) {}

    enum class State {
      // The stream is not finished and the worker is heartbeating.
      kActive,
      // The stream is finished.
      kDone,
    };

    // A counter of assigned splits for each source.
    std::vector<int64_t> num_assigned_splits_per_source;

    int64_t num_assigned_splits() const {
      return absl::c_accumulate(num_assigned_splits_per_source, 0);
    }

    State state = State::kActive;
  };

  struct Source {
    // A split provider for each input source of the dataset being snapshotted.
    std::unique_ptr<SplitProvider> split_provider;
    // The number of times the split provider has repeated.
    int64_t repetition_index = 0;
  };

  std::vector<Source> sources_;
  // Creates sources for the specified dataset.
  StatusOr<std::vector<Source>> CreateSources(
      const DatasetDef& dataset_def) const;
  // Resets a source when it runs out of splits, to support repetitions.
  Status ResetSource(Source& source, int64_t source_index);
  int64_t num_sources() const { return sources_.size(); }

  // All streams for this snapshot.
  std::vector<Stream> streams_;
  // A mapping of assigned worker to stream index.
  absl::flat_hash_map<std::string, int64_t> assignments_;
  // A counter of completed streams for this snapshot.
  int64_t num_completed_streams_ = 0;

  // A counter of assigned splits for this snapshot.
  int64_t num_assigned_splits_ = 0;
  // A counter of completed splits for this snapshot.
  int64_t num_completed_splits_ = 0;

  enum class Mode {
    // No streams are done.
    kActive,
    // At least one source is fully processed, but not all streams are done.
    kWindingDown,
    // All streams are done.
    kDone,
    // If any stream fails, the snapshot is in an error state. `status_` will
    // contain the error status.
    kError,
  };

  // If not `kActive`, at least one source has finished processing and no new
  // streams are created or assigned.
  Mode mode_ = Mode::kActive;

  // If `mode_` is in an error state, `status_` will contain the error status.
  Status status_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_MANAGER_H_
