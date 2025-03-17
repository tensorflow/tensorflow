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

#include "tensorflow/core/data/service/snapshot/snapshot_manager.h"

#include <algorithm>
#include <cstddef>
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
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/io/compression.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/prefetched_split_provider.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/path.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace {

const absl::Duration kProgressLoggingInterval = absl::Minutes(1);

absl::StatusOr<int64_t> CountSplits(SplitProvider& split_provider) {
  if (split_provider.Cardinality() != kUnknownCardinality) {
    return split_provider.Cardinality();
  }

  int64_t num_splits = 0;
  Tensor tensor;
  for (bool end_of_splits = false; !end_of_splits; ++num_splits) {
    TF_RETURN_IF_ERROR(split_provider.GetNext(&tensor, &end_of_splits));
  }
  --num_splits;
  TF_RETURN_IF_ERROR(split_provider.Reset());
  return num_splits;
}

absl::Status SkipSplit(SplitProvider& split_provider,
                       int64_t& repetition_index) {
  Tensor tensor;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider.GetNext(&tensor, &end_of_splits));
  while (end_of_splits) {
    ++repetition_index;
    TF_RETURN_IF_ERROR(split_provider.Reset());
    TF_RETURN_IF_ERROR(split_provider.GetNext(&tensor, &end_of_splits));
  }
  return absl::OkStatus();
}

std::string PrefetchedSplitDir(const std::string& snapshot_path,
                               int64_t source_index) {
  return tsl::io::JoinPath(snapshot_path, "prefetched_splits",
                           absl::StrCat("source_", source_index));
}

}  // namespace

absl::StatusOr<bool> SnapshotAssignmentManager::TryAddAssignment(
    absl::string_view snapshot_path, absl::string_view worker_address,
    int64_t stream_index) TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (assignments_[worker_address].size() >=
      worker_max_concurrent_snapshots()) {
    return false;
  }
  Assignment assignment{std::string(snapshot_path), stream_index};
  auto [unused, success] = assignments_[worker_address].insert(assignment);
  if (!success) {
    return absl::InternalError(absl::StrCat("Worker ", worker_address,
                                            " already had an assignment for ",
                                            assignment.DebugString()));
  }
  ++snapshot_assignment_counts_[snapshot_path];
  return true;
}

void SnapshotAssignmentManager::RemoveAssignment(
    absl::string_view snapshot_path, absl::string_view worker_address,
    int64_t stream_index) TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  auto num_erased = assignments_[worker_address].erase(
      {std::string(snapshot_path), stream_index});
  if ((snapshot_assignment_counts_[snapshot_path] -= num_erased) <= 0) {
    snapshot_assignment_counts_.erase(snapshot_path);
  }
}

void SnapshotAssignmentManager::AddSnapshot(absl::string_view snapshot_path)
    TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!snapshot_assignment_counts_.contains(snapshot_path)) {
    snapshot_assignment_counts_[snapshot_path] = 0;
  }
}

std::vector<std::string> SnapshotAssignmentManager::LoadBalanceSnapshots(
    absl::string_view worker_address) TF_LOCKS_EXCLUDED(mu_) {
  std::vector<std::string> result;

  tsl::mutex_lock l(mu_);
  result.reserve(snapshot_assignment_counts_.size());
  const auto it = assignments_.find(worker_address);
  if (it != assignments_.end()) {
    for (const Assignment& assignment : it->second) {
      result.push_back(assignment.snapshot_path);
    }
  }
  if (result.size() >= worker_max_concurrent_snapshots()) {
    return result;
  }

  absl::btree_multimap<size_t, std::string> snapshots_by_count;
  for (const auto& [snapshot, count] : snapshot_assignment_counts_) {
    snapshots_by_count.emplace(count, snapshot);
  }

  for (const auto& [_, snapshot] : snapshots_by_count) {
    if (absl::c_find(result, snapshot) == result.end()) {
      // Assigns the next least-assigned snapshot. Assigns one snapshot at a
      // time in case workers reach the assignment limit before the user has
      // submitted all requests.
      result.push_back(snapshot);
      return result;
    }
  }
  return result;
}

absl::StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Start(
    const SnapshotRequest& request,
    SnapshotAssignmentManager& assignment_manager, Env* env) {
  std::unique_ptr<SnapshotManager> snapshot_manager{
      new SnapshotManager{request.path(), assignment_manager, env}};
  TF_RETURN_IF_ERROR(snapshot_manager->Start(request));
  return snapshot_manager;
}

absl::Status SnapshotManager::Start(const SnapshotRequest& request)
    TF_LOCKS_EXCLUDED(mu_) {
  LOG(INFO) << "Starting to write tf.data snapshot at " << request.path();
  if (env_->FileExists(request.path()).ok()) {
    return errors::AlreadyExists("tf.data snapshot at ", request.path(),
                                 " already exists.");
  }
  tsl::mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(WriteOnDiskSkeleton());
  TF_RETURN_IF_ERROR(WriteOnDiskMetadata(request));
  TF_ASSIGN_OR_RETURN(sources_, CreateSources(request.dataset()));
  TF_ASSIGN_OR_RETURN(num_total_splits_, GetSplitsCardinality());
  metadata_ = request.metadata();
  LOG(INFO) << "Started writing tf.data distributed snapshot at " << path_;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<SnapshotManager::Source>>
SnapshotManager::CreateSources(const DatasetDef& dataset_def) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(CreateSplitProviders(dataset_def, split_providers));
  std::vector<SnapshotManager::Source> sources;
  sources.reserve(split_providers.size());
  for (size_t i = 0; i < split_providers.size(); ++i) {
    TF_ASSIGN_OR_RETURN(size_t cardinality, CountSplits(*split_providers[i]));
    sources.emplace_back(
        std::make_unique<PrefetchedSplitProvider>(
            std::move(split_providers[i]), PrefetchedSplitDir(path_, i), env_),
        /*repetition_index=*/0, cardinality);
  }
  return sources;
}

absl::StatusOr<int64_t> SnapshotManager::GetSplitsCardinality()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return absl::c_accumulate(sources_, 0,
                            [](size_t cardinality, const Source& source) {
                              return cardinality + source.cardinality;
                            });
}

absl::Status SnapshotManager::WriteOnDiskSkeleton()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(
      env_->RecursivelyCreateDir(CommittedChunksDirectory(path_)));
  TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(StreamsDirectory(path_)));
  return absl::OkStatus();
}

absl::Status SnapshotManager::WriteOnDiskMetadata(
    const SnapshotRequest& request) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(AtomicallyWriteTextProto(SnapshotMetadataFilePath(path_),
                                              request.metadata(), env_));
  TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(
      DatasetSpecFilePath(path_), request.metadata().element_spec(), env_));
  TF_RETURN_IF_ERROR(AtomicallyWriteBinaryProto(DatasetDefFilePath(path_),
                                                request.dataset(), env_));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Resume(
    absl::string_view path, SnapshotAssignmentManager& assignment_manager,
    Env* env) {
  SnapshotManager* snapshot_manager =
      new SnapshotManager(path, assignment_manager, env);
  TF_RETURN_IF_ERROR(snapshot_manager->Resume());
  return absl::WrapUnique(snapshot_manager);
}

absl::Status SnapshotManager::Resume() TF_LOCKS_EXCLUDED(mu_) {
  tsl::mutex_lock l(mu_);
  if (!env_->FileExists(path_).ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to recover tf.data snapshot at ", path_,
                     ": the snapshot path doesn't exist."));
  }
  if (env_->FileExists(SnapshotDoneFilePath(path_)).ok()) {
    mode_ = Mode::kDone;
    LOG(INFO) << "Recovered finished tf.data snapshot at " << path_;
    return absl::OkStatus();
  }
  if (env_->FileExists(SnapshotErrorFilePath(path_)).ok()) {
    mode_ = Mode::kError;
    StatusProto status_proto;
    TF_RETURN_IF_ERROR(
        ReadTextProto(env_, SnapshotErrorFilePath(path_), &status_proto));
    status_ = tsl::StatusFromProto(status_proto);
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(ReadOnDiskMetadata());
  TF_RETURN_IF_ERROR(ReadOnDiskStreams());
  LOG(INFO) << "Resumed writing tf.data distributed snapshot at " << path_;
  return absl::OkStatus();
}

absl::Status SnapshotManager::ReadOnDiskMetadata()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!env_->FileExists(SnapshotMetadataFilePath(path_)).ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to recover snapshot at ", path_,
                     ": snapshot has no snapshot.metadata"));
  }
  TF_RETURN_IF_ERROR(
      ReadTextProto(env_, SnapshotMetadataFilePath(path_), &metadata_));

  if (!env_->FileExists(DatasetDefFilePath(path_)).ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to recovery snapshot at ", path_,
                     ": snapshot has no dataset_def.proto"));
  }
  return absl::OkStatus();
}

// TODO(yangchen): Refactor this method.
absl::Status SnapshotManager::ReadOnDiskStreams()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string streams_path = StreamsDirectory(path_);
  TF_ASSIGN_OR_RETURN(const std::vector<std::string> stream_directories,
                      GetChildren(streams_path, env_));

  DatasetDef dataset_def;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(env_, DatasetDefFilePath(path_), &dataset_def));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(CreateSplitProviders(dataset_def, split_providers));
  std::vector<int64_t> repetition_indices(split_providers.size(), 0);
  std::vector<int64_t> cardinalities;
  for (size_t i = 0; i < split_providers.size(); ++i) {
    TF_ASSIGN_OR_RETURN(int64_t cardinality, CountSplits(*split_providers[i]));
    cardinalities.push_back(cardinality);
  }

  tsl::mutex mu;  // Protects `resume_status` and `global_split_indices`.
  absl::Status resume_status;
  absl::flat_hash_set<int64_t> global_split_indices;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      env_, tsl::ThreadOptions{}, "restore_snapshot_stream_thread",
      std::max(size_t{1}, stream_directories.size()));
  for (const auto& stream_directory : stream_directories) {
    std::string stream_path = tsl::io::JoinPath(streams_path, stream_directory);

    // `stream_directory` must have this format: "stream_<stream_index>".
    std::vector<std::string> tokens = absl::StrSplit(stream_directory, '_');
    int64_t stream_index;
    if (tokens.size() != 2 || !absl::SimpleAtoi(tokens[1], &stream_index) ||
        stream_index < 0) {
      return absl::InternalError(absl::StrCat(
          "Can't parse tf.data snapshot stream directory ", stream_path,
          ": filename must have the format stream_<stream_index>."));
    }

    thread_pool->Schedule([this, &stream_directories, stream_index,
                           &split_providers, &repetition_indices,
                           &global_split_indices, &resume_status,
                           &mu]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      StreamRestorer stream_restorer(env_, path_, stream_index,
                                     split_providers.size(),
                                     assignment_manager_);
      absl::Status s = stream_restorer.ReadOnDiskStream();
      tsl::mutex_lock l(mu);
      resume_status.Update(s);
      resume_status.Update(RestoreFrom(stream_restorer, stream_directories,
                                       split_providers, repetition_indices,
                                       global_split_indices));
    });
  }
  thread_pool.reset();
  TF_RETURN_IF_ERROR(resume_status);

  for (int64_t i = 0; i < split_providers.size(); ++i) {
    sources_.emplace_back(
        std::make_unique<PrefetchedSplitProvider>(
            std::move(split_providers[i]), PrefetchedSplitDir(path_, i), env_),
        repetition_indices[i], cardinalities[i]);
  }
  TF_ASSIGN_OR_RETURN(num_total_splits_, GetSplitsCardinality());

  for (int64_t i = 0; i < global_split_indices.size(); ++i) {
    if (!global_split_indices.contains(i)) {
      return absl::InternalError(
          absl::StrCat("Failed to restore tf.data snapshot at ", path_,
                       ": Found missing global split index ", i, "."));
    }
  }
  num_assigned_splits_ = global_split_indices.size();

  if (!streams_.empty() && absl::c_all_of(streams_, [](const auto& stream) {
        return stream.second.state == Stream::State::kDone;
      })) {
    mode_ = Mode::kDone;
    TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(SnapshotDoneFilePath(path_),
                                                   std::string(), env_));
    LOG(INFO) << "Finished writing tf.data distributed snapshot at " << path_;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string>
SnapshotManager::StreamRestorer::OwnerWorkerAddress() const {
  std::string worker_address;
  TF_RETURN_IF_ERROR(
      env_->FileExists(StreamWorkerFilePath(path_, stream_index_)));
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      env_, StreamWorkerFilePath(path_, stream_index_), &worker_address));
  return worker_address;
}

absl::Status SnapshotManager::StreamRestorer::ReadOnDiskStream() {
  absl::StatusOr<std::string> worker_address = OwnerWorkerAddress();
  if (!worker_address.ok()) {
    // This could happen if the dispatcher fails after creating a stream
    // directory before writing the owner file. The snapshot manager can check
    // this case by testing if GetStream() returns a value.
    return absl::OkStatus();
  }

  worker_address_ = *worker_address;
  restored_stream_.emplace(num_sources_);
  std::string splits_path = SplitsDirectory(path_, stream_index_);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> source_directories,
                      GetChildren(splits_path, env_));

  for (const auto& source_directory : source_directories) {
    std::string source_path = tsl::io::JoinPath(splits_path, source_directory);

    // `source_directory` must have this format: "source_<source_index>".
    std::vector<std::string> tokens = absl::StrSplit(source_directory, '_');
    int64_t source_index = 0;
    if (tokens.size() != 2 || !absl::SimpleAtoi(tokens[1], &source_index) ||
        source_index < 0) {
      return absl::InternalError(absl::StrCat(
          "Can't parse tf.data snapshot source directory ", source_path,
          ": filename must have the format source_<source_index>."));
    }
    if (source_index >= num_sources_) {
      return absl::InternalError(
          absl::StrCat("Found conflict between the number of sources, ",
                       num_sources_, ", and the filename of ", source_path));
    }
    TF_RETURN_IF_ERROR(ReadOnDiskSource(source_index));
  }

  if (env_->FileExists(StreamDoneFilePath(path_, stream_index_)).ok()) {
    restored_stream_->state = Stream::State::kDone;
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(bool assignment_added,
                      assignment_manager_.TryAddAssignment(
                          path_, *worker_address, stream_index_));
  if (!assignment_added) {
    return absl::InternalError(absl::StrCat(
        "Failed to recover tf.data snapshot dispatcher: Worker ",
        *worker_address, " was assigned too many streams. At most ",
        assignment_manager_.worker_max_concurrent_snapshots(),
        " streams are allowed."));
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::StreamRestorer::ReadOnDiskSource(
    int64_t source_index) {
  std::string source_directory =
      SourceDirectory(path_, stream_index_, source_index);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> repetition_directories,
                      GetChildren(source_directory, env_));

  for (const std::string& repetition : repetition_directories) {
    std::string repetition_dir =
        tsl::io::JoinPath(source_directory, repetition);
    TF_ASSIGN_OR_RETURN(std::vector<std::string> split_files,
                        GetChildren(repetition_dir, env_));
    for (const std::string& split_file : split_files) {
      std::string split_path = tsl::io::JoinPath(repetition_dir, split_file);
      TF_RETURN_IF_ERROR(
          ReadOnDiskSplit(source_index, split_files, split_path));
    }
    restored_stream_->num_assigned_splits_per_source[source_index] +=
        split_files.size();
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::StreamRestorer::ReadOnDiskSplit(
    int64_t source_index, const std::vector<std::string>& split_files,
    const std::string& split_file) {
  // `split_file` must have this format:
  // "split_<local_split_index>_<global_split_index>".
  TF_ASSIGN_OR_RETURN(auto split_indices, ParseSplitFilename(split_file));
  auto [local_split_index, global_split_index] = split_indices;
  if (global_split_indices_.contains(global_split_index)) {
    return absl::InternalError(absl::StrCat(
        "Failed to restore tf.data snapshot at ", path_,
        ": Found duplicate global split index in split ", split_file, "."));
  }
  global_split_indices_.insert(global_split_index);
  return absl::OkStatus();
}

absl::Status SnapshotManager::RestoreFrom(
    const StreamRestorer& stream_restorer,
    const std::vector<std::string>& stream_directories,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers,
    std::vector<std::int64_t>& repetition_indices,
    absl::flat_hash_set<int64_t>& global_split_indices)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!stream_restorer.GetStream().has_value()) {
    // The dispatcher may get preempted when it writes the owner_worker file.
    // If that happens, we skip the stream directory.
    return absl::OkStatus();
  }

  streams_.insert(
      {stream_restorer.StreamIndex(), *stream_restorer.GetStream()});
  auto [it, success] = assignments_.insert(
      {stream_restorer.WorkerAddress(), stream_restorer.StreamIndex()});
  if (!success) {
    return absl::InternalError(absl::StrCat(
        "tf.data dispatcher failed to assign stream ",
        stream_restorer.StreamIndex(), " to snapshot worker ",
        stream_restorer.WorkerAddress(),
        ": The worker is already assigned stream ", it->second, "."));
  }
  for (int64_t source_index = 0; source_index < repetition_indices.size();
       ++source_index) {
    // To account for the splits having been assigned, skips the splits in the
    // respective split providers.
    int64_t skip_splits = GetStream(stream_restorer.StreamIndex())
                              .num_assigned_splits_per_source[source_index];
    for (int64_t i = 0; i < skip_splits; ++i) {
      TF_RETURN_IF_ERROR(SkipSplit(*split_providers[source_index],
                                   repetition_indices[source_index]));
    }
  }
  for (int64_t global_split_index : stream_restorer.GlobalSplitIndices()) {
    if (global_split_indices.contains(global_split_index)) {
      return absl::InternalError(
          absl::StrCat("Failed to restore tf.data snapshot at ", path_,
                       ": Found ", "duplicate global split index in stream ",
                       stream_restorer.StreamIndex(), "."));
    }
    global_split_indices.insert(global_split_index);
  }
  return absl::OkStatus();
}

SnapshotManager::Stream& SnapshotManager::GetStream(int64_t stream_index)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  auto [it, _] = streams_.try_emplace(stream_index, num_sources());
  return it->second;
}

absl::Status SnapshotManager::HandleStreamCompletion(
    int64_t stream_index, absl::string_view worker_address)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  GetStream(stream_index).state = Stream::State::kDone;
  assignment_manager_.RemoveAssignment(path_, worker_address, stream_index);
  ++num_completed_streams_;
  if (absl::c_all_of(streams_, [](const auto& stream) {
        return stream.second.state == Stream::State::kDone;
      })) {
    mode_ = Mode::kDone;
    TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(SnapshotDoneFilePath(path_),
                                                   std::string(), env_));
    LOG(INFO) << "Finished writing tf.data distributed snapshot at " << path_;
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::HandleStreamError(
    absl::string_view worker_address, const StatusProto& status_proto)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // This method returns an OkStatus as the RPC status if the worker reports an
  // error. The errors are communicated back to the workers with a proper RPC
  // response, instead of with a error status.
  if (!status_.ok()) {
    return absl::OkStatus();
  }

  mode_ = Mode::kError;
  status_ = tsl::StatusFromProto(status_proto);
  TF_RETURN_IF_ERROR(AtomicallyWriteTextProto(SnapshotErrorFilePath(path_),
                                              status_proto, env_));
  LOG(ERROR) << "Failed to write tf.data distributed snapshot at " << path_
             << ". Worker " << worker_address << " reported error: " << status_;
  return absl::OkStatus();
}

absl::StatusOr<std::optional<int64_t>>
SnapshotManager::MaybeCreateAndAssignNewStream(absl::string_view worker_address)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t new_stream_index =
      streams_.empty() ? 0 : streams_.rbegin()->first + 1;
  TF_ASSIGN_OR_RETURN(bool assignment_added,
                      assignment_manager_.TryAddAssignment(
                          path_, worker_address, new_stream_index));
  if (!assignment_added) {
    return std::optional<int64_t>();
  }
  streams_.insert({new_stream_index, Stream(num_sources())});
  assignments_[worker_address] = new_stream_index;
  return new_stream_index;
}

absl::StatusOr<std::optional<std::pair<int64_t, bool>>>
SnapshotManager::MaybeGetOrCreateStreamAssignment(
    absl::string_view worker_address,
    const SnapshotTaskProgress* snapshot_progress)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::optional<int64_t> assigned_stream_index;
  if (auto it = assignments_.find(worker_address); it != assignments_.end()) {
    assigned_stream_index = it->second;
  }
  if (snapshot_progress) {
    if (assigned_stream_index.has_value() &&
        *assigned_stream_index !=
            snapshot_progress->snapshot_task().stream_index()) {
      return absl::InternalError(absl::StrCat(
          "tf.data snapshot worker ", worker_address, " was assigned stream ",
          snapshot_progress->snapshot_task().stream_index(),
          ", but is now assigned a different stream ", *assigned_stream_index));
    }
    if (assigned_stream_index.has_value() && snapshot_progress->completed()) {
      TF_RETURN_IF_ERROR(HandleStreamCompletion(
          snapshot_progress->snapshot_task().stream_index(), worker_address));
      return std::nullopt;
    }
    if (snapshot_progress->status().code() != error::OK) {
      TF_RETURN_IF_ERROR(
          HandleStreamError(worker_address, snapshot_progress->status()));
      return std::nullopt;
    }
  }
  if (!assigned_stream_index) {
    if (mode_ != Mode::kActive) {
      return std::nullopt;
    }
    TF_ASSIGN_OR_RETURN(assigned_stream_index,
                        MaybeCreateAndAssignNewStream(worker_address));
    if (!assigned_stream_index.has_value()) {
      return std::nullopt;
    }
    return std::make_pair(*assigned_stream_index, true);
  }
  if (!assigned_stream_index.has_value() ||
      GetStream(*assigned_stream_index).state == Stream::State::kDone) {
    return std::nullopt;
  }
  return std::make_pair(*assigned_stream_index, false);
}

absl::Status SnapshotManager::WorkerHeartbeat(
    const WorkerHeartbeatRequest& request, WorkerHeartbeatResponse& response)
    TF_LOCKS_EXCLUDED(mu_) {
  std::optional<std::pair<int64_t, bool>> assigned_stream_index;
  std::vector<int64_t> repetitions_per_source;

  {
    tsl::mutex_lock l(mu_);
    dead_workers_.erase(request.worker_address());

    if (mode_ == Mode::kDone || mode_ == Mode::kError) {
      // When the snapshot manager is done or in an error state, it returns an
      // empty response to inform the workers to cancel the ongoing tasks.
      return absl::OkStatus();
    }

    if (absl::Time now = absl::FromUnixMicros(env_->NowMicros());
        now - last_progress_log_time_ > kProgressLoggingInterval) {
      LOG(INFO) << "tf.data snapshot progress [" << path_
                << "]: " << num_completed_streams_ << "/" << streams_.size()
                << " streams completed; " << num_assigned_splits_ << "/"
                << num_total_splits_ << " splits assigned or completed.";
      last_progress_log_time_ = now;
    }

    const SnapshotTaskProgress* snapshot_progress = nullptr;
    if (auto it = request.snapshot_task_progress().find(path_);
        it != request.snapshot_task_progress().end()) {
      snapshot_progress = &it->second;
    }
    if (snapshot_progress && snapshot_progress->completed() &&
        mode_ == Mode::kActive) {
      mode_ = Mode::kWindingDown;
    }
    TF_ASSIGN_OR_RETURN(assigned_stream_index,
                        MaybeGetOrCreateStreamAssignment(
                            request.worker_address(), snapshot_progress));
    if (!assigned_stream_index.has_value()) {
      return absl::OkStatus();
    }

    SnapshotTaskDef* snapshot_task = response.add_snapshot_tasks();
    snapshot_task->set_base_path(path_);
    snapshot_task->set_num_sources(num_sources());
    *snapshot_task->mutable_metadata() = metadata_;
    snapshot_task->set_stream_index(assigned_stream_index->first);

    for (int64_t source_index = 0; source_index < num_sources();
         ++source_index) {
      repetitions_per_source.push_back(sources_[source_index].repetition_index);
    }
  }  // Releases `mu_`.

  const auto [stream_index, is_new_stream] = *assigned_stream_index;
  if (is_new_stream) {
    TF_RETURN_IF_ERROR(InitStreamDirectory(
        stream_index, request.worker_address(), repetitions_per_source));
    LOG(INFO) << "For snapshot at " << path_ << ", created stream_"
              << stream_index << " and assigned to "
              << request.worker_address();
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::InitStreamDirectory(
    int64_t stream_index, const std::string& worker_address,
    const std::vector<int64_t>& repetitions_per_source) {
  for (int64_t source_index = 0; source_index < repetitions_per_source.size();
       ++source_index) {
    for (int64_t repetition_index = 0;
         repetition_index <= repetitions_per_source[source_index];
         ++repetition_index) {
      TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(RepetitionDirectory(
          path_, stream_index, source_index, repetition_index)));
    }
  }
  return AtomicallyWriteStringToFile(StreamWorkerFilePath(path_, stream_index),
                                     worker_address, env_);
}

absl::Status SnapshotManager::GetSnapshotSplit(
    const GetSnapshotSplitRequest& request, GetSnapshotSplitResponse& response)
    TF_LOCKS_EXCLUDED(get_split_mu_, mu_) {
  int64_t local_split_index = 0;
  int64_t global_split_index = 0;
  PrefetchedSplitProvider* split_provider = nullptr;
  tsl::mutex_lock get_split_lock(get_split_mu_);
  {
    tsl::mutex_lock l(mu_);
    if (auto it = assignments_.find(request.worker_address());
        it == assignments_.end()) {
      return absl::InternalError(
          absl::StrCat("tf.data snapshot worker ", request.worker_address(),
                       " was assigned stream ", request.stream_index(),
                       ", but the assignment is no longer available."));
    } else if (it->second != request.stream_index()) {
      return absl::InternalError(
          absl::StrCat("tf.data snapshot worker ", request.worker_address(),
                       " was assigned stream ", request.stream_index(),
                       " but is now assigned a different stream ", it->second));
    }

    Stream& stream = GetStream(request.stream_index());
    local_split_index =
        stream.num_assigned_splits_per_source[request.source_index()];
    global_split_index = num_assigned_splits_;
    response.set_local_split_index(local_split_index);

    Source& source = sources_[request.source_index()];
    if (request.repetition_index() < source.repetition_index) {
      response.set_end_of_splits(true);
      return absl::OkStatus();
    }
    while (request.repetition_index() > source.repetition_index) {
      // This could happen if an iterator is repeated before reaching end of
      // input, e.g. for the longer input to `Dataset.zip`. In this case we mark
      // the previous repetitions as completed and advance to the requested
      // repetition.
      TF_RETURN_IF_ERROR(ResetSource(source, request.source_index()));
    }
    split_provider = source.split_provider.get();
  }

  std::string split_path = SplitPath(
      path_, request.stream_index(), request.source_index(),
      request.repetition_index(), local_split_index, global_split_index);
  TF_ASSIGN_OR_RETURN(std::optional<Tensor> split,
                      split_provider->GetNext(split_path));
  if (!split.has_value()) {
    response.set_end_of_splits(true);
    return absl::OkStatus();
  }
  split->AsProtoTensorContent(response.mutable_split());

  tsl::mutex_lock l(mu_);
  ++GetStream(request.stream_index())
        .num_assigned_splits_per_source[request.source_index()];
  ++num_assigned_splits_;
  return absl::OkStatus();
}

absl::Status SnapshotManager::ResetSource(Source& source, int64_t source_index)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(source.split_provider->Reset());
  ++source.repetition_index;
  LOG(INFO) << "Starting repetition_" << source.repetition_index << " "
            << "for snapshot " << path_ << ", source " << source_index;
  for (const auto& [stream_index, _] : streams_) {
    TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(RepetitionDirectory(
        path_, stream_index, source_index, source.repetition_index)));
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::GetSnapshotStreams(
    GetSnapshotStreamsResponse& response) TF_LOCKS_EXCLUDED(mu_) {
  tsl::tf_shared_lock l(mu_);
  for (const auto& [stream_index, stream] : streams_) {
    SnapshotStreamInfo* stream_info = response.add_streams();
    stream_info->set_index(stream_index);
    stream_info->set_state(stream.state == Stream::State::kDone
                               ? SnapshotStreamInfo::DONE
                               : SnapshotStreamInfo::ASSIGNED);
  }
  return absl::OkStatus();
}

void SnapshotManager::Cancel() {
  std::vector<PrefetchedSplitProvider*> split_providers_to_cancel;
  {
    tsl::mutex_lock l(mu_);
    for (Source& source : sources_) {
      split_providers_to_cancel.push_back(source.split_provider.get());
    }
  }

  for (PrefetchedSplitProvider* split_provider : split_providers_to_cancel) {
    split_provider->Cancel();
  }
}

}  // namespace data
}  // namespace tensorflow
