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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {

// A helper shared among `SnapshotManager`s to limit workers' stream assignments
// across ongoing snapshots.
class SnapshotAssignmentManager {
 public:
  explicit SnapshotAssignmentManager(int64_t worker_max_concurrent_snapshots)
      : worker_max_concurrent_snapshots_(worker_max_concurrent_snapshots) {}

  // Tries to record the event of a worker being assigned a stream. Returns
  // `false` if the worker has too many assignments. Returns an error if the
  // worker is already known to have been assigned this stream.
  absl::StatusOr<bool> TryAddAssignment(absl::string_view snapshot_path,
                                        absl::string_view worker_address,
                                        int64_t stream_index);

  // Records the event of a worker stopping work on a stream.
  void RemoveAssignment(absl::string_view snapshot_path,
                        absl::string_view worker_address, int64_t stream_index);

  // Returns the maximum concurrent snapshots processed by each worker.
  int64_t worker_max_concurrent_snapshots() const {
    return worker_max_concurrent_snapshots_;
  }

 private:
  struct Assignment {
    std::string snapshot_path;
    int64_t stream_index;

    template <typename H>
    friend H AbslHashValue(H h, const Assignment& a) {
      return H::combine(std::move(h), a.snapshot_path, a.stream_index);
    }

    friend bool operator==(const Assignment& lhs, const Assignment& rhs) {
      return lhs.snapshot_path == rhs.snapshot_path &&
             lhs.stream_index == rhs.stream_index;
    }

    std::string DebugString() const {
      return absl::Substitute(
          "Assignment { snapshot_path: $0, stream_index: $1 }", snapshot_path,
          stream_index);
    }
  };

  // A mapping of worker address to ongoing assignments.
  absl::flat_hash_map<std::string, absl::flat_hash_set<Assignment>> assignments_
      TF_GUARDED_BY(mu_);

  // The maximum number of snapshots that a worker can concurrently process at a
  // given point in time. This is a tradeoff between worker resource usage and
  // snapshot wall time. A value of 0 indicates that the decision should be left
  // up to the runtime.
  const int64_t worker_max_concurrent_snapshots_;

  mutable tsl::mutex mu_;
};

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
  static absl::StatusOr<std::unique_ptr<SnapshotManager>> Start(
      const SnapshotRequest& request,
      SnapshotAssignmentManager& assignment_manager, Env* env);
  // Resumes an existing snapshot process, reading from the on-disk state in
  // `path` to derive an in-memory state. Returns an error if `path` is in a bad
  // state.
  static absl::StatusOr<std::unique_ptr<SnapshotManager>> Resume(
      absl::string_view path, SnapshotAssignmentManager& assignment_manager,
      Env* env);

  // Handles the work pertaining to this snapshot process for the respective
  // `DispatcherService` API calls:
  // - `WorkerHeartbeat`: Returns a stream assignment for the worker.
  // - `GetSnapshotSplit`: Returns a split assignment for the worker.
  // - `GetSnapshotStreams`: Returns information about all streams.
  absl::Status WorkerHeartbeat(const WorkerHeartbeatRequest& request,
                               WorkerHeartbeatResponse& response);
  absl::Status GetSnapshotSplit(const GetSnapshotSplitRequest& request,
                                GetSnapshotSplitResponse& response);
  absl::Status GetSnapshotStreams(GetSnapshotStreamsResponse& response);

 private:
  SnapshotManager(absl::string_view path,
                  SnapshotAssignmentManager& assignment_manager, Env* env)
      : path_(path),
        env_(env),
        last_progress_log_time_(absl::FromUnixMicros(env->NowMicros())),
        assignment_manager_(assignment_manager) {}

  // Helpers for `Start` above. These update the on-disk state.
  absl::Status Start(const SnapshotRequest& request);
  absl::Status WriteOnDiskSkeleton();
  absl::Status WriteOnDiskMetadata(const SnapshotRequest& request);

  // Helpers for `Resume` above. These update the in-memory state.
  absl::Status Resume();
  absl::Status ReadOnDiskMetadata();
  absl::Status ReadOnDiskStreams();
  absl::StatusOr<std::string> OwnerWorkerAddress(
      const std::string& stream_directory) const;
  absl::Status ReadOnDiskStream(
      int64_t stream_index, const std::string& worker_address,
      absl::flat_hash_set<int64_t>& global_split_indices);
  absl::Status ReadOnDiskSource(
      int64_t stream_index, int64_t source_index,
      absl::flat_hash_set<int64_t>& global_split_indices);
  absl::Status ReadOnDiskSplit(
      int64_t source_index, const std::vector<std::string>& split_files,
      std::string split_file,
      absl::flat_hash_set<int64_t>& global_split_indices);
  absl::Status RecoverSplit(const std::string& temp_split_file,
                            SplitProvider& split_provider);
  absl::StatusOr<Tensor> GetNextSplit(SplitProvider& split_provider);

  // Helpers for `WorkerHeartbeat` above. These may update the in-memory and
  // on-disk states.
  absl::StatusOr<std::optional<int64_t>> MaybeGetOrCreateStreamAssignment(
      absl::string_view worker_address,
      const SnapshotTaskProgress* snapshot_progress);
  absl::Status HandleStreamCompletion(int64_t stream_index,
                                      absl::string_view worker_address);
  void ReassignPreviouslyAssignedStream(int64_t stream_index,
                                        absl::string_view worker_address);
  std::optional<int64_t> MaybeAssignOrphanStream(
      absl::string_view worker_address);
  absl::StatusOr<std::optional<int64_t>> MaybeCreateAndAssignNewStream(
      absl::string_view worker_address);
  absl::Status HandleStreamError(absl::string_view worker_address,
                                 const StatusProto& status_proto);

  mutable tsl::mutex mu_;

  // The filepath of the on-disk state.
  const std::string path_;
  // A tensorflow environment interface used to write to and read from `path_`.
  tsl::Env* const env_;
  // Distributed snapshot metadata.
  experimental::DistributedSnapshotMetadata metadata_ TF_GUARDED_BY(mu_);
  // The last time progress was logged.
  absl::Time last_progress_log_time_ TF_GUARDED_BY(mu_);

  // The addresses of all workers considered to be dead based on heartbeat
  // timeout.
  absl::flat_hash_set<std::string> dead_workers_ TF_GUARDED_BY(mu_);

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

  std::vector<Source> sources_ TF_GUARDED_BY(mu_);
  // Creates sources for the specified dataset.
  absl::StatusOr<std::vector<Source>> CreateSources(
      const DatasetDef& dataset_def) const;
  // Counts the number of splits for a single repetition of the data in
  // `sources_`.
  absl::StatusOr<int64_t> CountSplits();
  // Resets a source when it runs out of splits, to support repetitions.
  absl::Status ResetSource(Source& source, int64_t source_index);
  int64_t num_sources() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return sources_.size();
  }

  // All streams for this snapshot.
  std::vector<Stream> streams_ TF_GUARDED_BY(mu_);
  // A counter of completed streams for this snapshot.
  int64_t num_completed_streams_ TF_GUARDED_BY(mu_) = 0;

  // A mapping of worker to assigned stream index for this snapshot.
  absl::flat_hash_map<std::string, int64_t> assignments_ TF_GUARDED_BY(mu_);
  // A mapping of worker to assigned streams for all snapshots.
  SnapshotAssignmentManager& assignment_manager_ TF_GUARDED_BY(mu_);

  // A counter of assigned splits for this snapshot.
  int64_t num_assigned_splits_ TF_GUARDED_BY(mu_) = 0;
  // The number of splits in a single repetition of the data in `sources_`.
  int64_t num_total_splits_ TF_GUARDED_BY(mu_) = 0;

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
  Mode mode_ TF_GUARDED_BY(mu_) = Mode::kActive;

  // If `mode_` is in an error state, `status_` will contain the error status.
  absl::Status status_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_MANAGER_H_
