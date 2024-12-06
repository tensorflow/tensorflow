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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/prefetched_split_provider.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// A helper shared among `SnapshotManager`s to limit workers' stream assignments
// across ongoing snapshots. This class is thread-safe.
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

  // Adds a new snapshot.
  void AddSnapshot(absl::string_view snapshot_path);

  // Load balances snapshots by the number of assigned streams. Given a worker,
  // returns snapshots in the following order:
  // - Snapshots already assigned to this worker.
  // - Snapshots with the fewest assignments.
  std::vector<std::string> LoadBalanceSnapshots(
      absl::string_view worker_address);

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

  // A mapping from snapshot to the number of assigned workers.
  absl::flat_hash_map<std::string, int64_t> snapshot_assignment_counts_
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

  // Cancels the SnapshotManager and finishes in-progress threads.
  void Cancel();

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

  // Helpers for `WorkerHeartbeat` above. These may update the in-memory and
  // on-disk states.
  // Gets or creates a new stream. Returns the stream index and a bool value
  // indicating whether a new stream has been created. Returns `std::nullopt`
  // if there are no more streams to write or there is an error.
  absl::StatusOr<std::optional<std::pair<int64_t, bool>>>
  MaybeGetOrCreateStreamAssignment(
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
  // Uses a separate mutex for `GetSnapshotSplit` RPCs. `GetSnapshotSplit` uses
  // file IO and may be slow, which may slow down `WorkerHeartbeat` RPCs if they
  // share one mutex.
  mutable tsl::mutex get_split_mu_;

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
    Source(std::unique_ptr<PrefetchedSplitProvider> split_provider,
           int64_t repetition_index, int64_t cardinality)
        : split_provider(std::move(split_provider)),
          repetition_index(repetition_index),
          cardinality(cardinality) {}

    // A split provider for each input source of the dataset being snapshotted.
    std::unique_ptr<PrefetchedSplitProvider> split_provider;
    // The number of times the split provider has repeated.
    int64_t repetition_index = 0;
    // The number of splits in `split_provider`.
    const int64_t cardinality;
  };

  // Helper class to restore a stream. Multiple stream restorers are safe to run
  // in parallel. After it reads the on-disk stream, the client is responsible
  // to apply the data to actually restore its internal states.
  class StreamRestorer {
   public:
    explicit StreamRestorer(tsl::Env* env, absl::string_view path,
                            int64_t stream_index, int64_t num_sources,
                            SnapshotAssignmentManager& assignment_manager)
        : env_(env),
          path_(path),
          stream_index_(stream_index),
          num_sources_(num_sources),
          assignment_manager_(assignment_manager) {}

    // Reads snapshot stream from the files and collects data for restoration.
    absl::Status ReadOnDiskStream();

    // Accessors for collected data. Should be called *after* `ReadOnDiskStream`
    // is called.
    const std::optional<Stream>& GetStream() const { return restored_stream_; }
    int64_t StreamIndex() const { return stream_index_; }
    const std::string& WorkerAddress() const { return worker_address_; }
    const absl::flat_hash_set<int64_t>& GlobalSplitIndices() const {
      return global_split_indices_;
    }

   private:
    absl::StatusOr<std::string> OwnerWorkerAddress() const;
    absl::Status ReadOnDiskSource(int64_t source_index);
    absl::Status ReadOnDiskSplit(int64_t source_index,
                                 const std::vector<std::string>& split_files,
                                 const std::string& split_file);
    absl::Status SkipSplit(SplitProvider& split_provider);

    tsl::Env* const env_;
    const std::string path_;
    const int64_t stream_index_;
    const int64_t num_sources_;
    SnapshotAssignmentManager& assignment_manager_;

    std::string worker_address_;
    std::optional<Stream> restored_stream_;
    absl::flat_hash_set<int64_t> global_split_indices_;
  };

  // Applies the data collected by `stream_restorer` to actually restore the
  // snapshot manager.
  absl::Status RestoreFrom(
      const StreamRestorer& stream_restorer,
      const std::vector<std::string>& stream_directories,
      std::vector<std::unique_ptr<SplitProvider>>& split_providers,
      std::vector<int64_t>& repetition_indices,
      absl::flat_hash_set<int64_t>& global_split_indices);

  // Gets the snapshot stream.
  Stream& GetStream(int64_t stream_index);
  // Initializes the stream directory.
  absl::Status InitStreamDirectory(
      int64_t stream_index, const std::string& worker_address,
      const std::vector<int64_t>& repetitions_per_source);

  std::vector<Source> sources_ TF_GUARDED_BY(mu_);
  // Creates sources for the specified dataset.
  absl::StatusOr<std::vector<Source>> CreateSources(
      const DatasetDef& dataset_def) const;
  // Returns the total number of splits.
  absl::StatusOr<int64> GetSplitsCardinality();
  // Resets a source when it runs out of splits, to support repetitions.
  absl::Status ResetSource(Source& source, int64_t source_index);
  int64_t num_sources() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return sources_.size();
  }

  // All streams for this snapshot.
  absl::btree_map<int64_t, Stream> streams_ TF_GUARDED_BY(mu_);
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
