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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/lib/io/compression.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/protobuf/error_codes.pb.h"
#include "tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {

const absl::Duration kProgressLoggingInterval = absl::Minutes(1);
const absl::string_view kSplitFileCompression = tsl::io::compression::kNone;

absl::StatusOr<bool> SnapshotAssignmentManager::TryAddAssignment(
    absl::string_view snapshot_path, absl::string_view worker_address,
    int64_t stream_index) {
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
  return true;
}

void SnapshotAssignmentManager::RemoveAssignment(
    absl::string_view snapshot_path, absl::string_view worker_address,
    int64_t stream_index) {
  tsl::mutex_lock l(mu_);
  assignments_[worker_address].erase(
      {std::string(snapshot_path), stream_index});
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
  TF_ASSIGN_OR_RETURN(sources_, CreateSources(request.dataset()));
  TF_ASSIGN_OR_RETURN(num_total_splits_, CountSplits());
  TF_RETURN_IF_ERROR(WriteOnDiskSkeleton());
  TF_RETURN_IF_ERROR(WriteOnDiskMetadata(request));
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
  for (auto& split_provider : split_providers) {
    sources.push_back({std::move(split_provider), /*repetition_index=*/0});
  }
  return sources;
}

absl::StatusOr<int64_t> SnapshotManager::CountSplits()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t num_splits = 0;
  for (const auto& source : sources_) {
    Tensor tensor;
    for (bool end_of_splits = false; !end_of_splits; ++num_splits) {
      TF_RETURN_IF_ERROR(
          source.split_provider->GetNext(&tensor, &end_of_splits));
    }
    --num_splits;
    TF_RETURN_IF_ERROR(source.split_provider->Reset());
  }
  return num_splits;
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
  TF_RETURN_IF_ERROR(WriteTextProto(env_, SnapshotMetadataFilePath(path_),
                                    request.metadata()));
  TF_RETURN_IF_ERROR(WriteStringToFile(env_, DatasetSpecFilePath(path_),
                                       request.metadata().element_spec()));
  TF_RETURN_IF_ERROR(
      WriteBinaryProto(env_, DatasetDefFilePath(path_), request.dataset()));
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
        absl::StrCat("Failed to recover snapshot at ", path_,
                     ": the snapshot path doesn't exist"));
  }
  if (env_->FileExists(SnapshotDoneFilePath(path_)).ok()) {
    mode_ = Mode::kDone;
    LOG(INFO) << "Attempted to recover snapshot at " << path_
              << " but it's already done";
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
  DatasetDef dataset_def;
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(env_, DatasetDefFilePath(path_), &dataset_def));

  TF_ASSIGN_OR_RETURN(sources_, CreateSources(dataset_def));
  TF_ASSIGN_OR_RETURN(num_total_splits_, CountSplits());
  return absl::OkStatus();
}

absl::Status SnapshotManager::ReadOnDiskStreams()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string streams_path = StreamsDirectory(path_);
  TF_ASSIGN_OR_RETURN(const std::vector<std::string> stream_directories,
                      GetChildren(streams_path, env_));
  streams_.resize(stream_directories.size(), Stream(num_sources()));

  absl::flat_hash_set<int64_t> global_split_indices;
  for (const auto& stream_directory : stream_directories) {
    std::string stream_path = tsl::io::JoinPath(streams_path, stream_directory);

    // `stream_directory` must have this format: "stream_<stream_index>".
    std::vector<std::string> tokens = absl::StrSplit(stream_directory, '_');
    int64_t stream_index;
    if (tokens.size() != 2 || !absl::SimpleAtoi(tokens[1], &stream_index) ||
        stream_index < 0) {
      return absl::InternalError(absl::StrCat(
          "Can't parse the name of ", stream_path,
          ": filename must have the format stream_<stream_index>."));
    }

    absl::StatusOr<std::string> worker_address =
        OwnerWorkerAddress(stream_path);
    if (!worker_address.ok()) {
      // The dispatcher may get preempted when it writes the owner_worker file.
      // If that happens, we skip the last stream directory.
      if (stream_index < stream_directories.size() - 1) {
        return worker_address.status();
      }
      streams_.pop_back();
      continue;
    }
    TF_RETURN_IF_ERROR(
        ReadOnDiskStream(stream_index, *worker_address, global_split_indices));
  }

  for (int64_t i = 0; i < global_split_indices.size(); ++i) {
    if (!global_split_indices.contains(i)) {
      return absl::InternalError(absl::StrCat(
          "Found missing global split index, ", i, ", in ", path_));
    }
  }
  num_assigned_splits_ = global_split_indices.size();

  if (!streams_.empty() && absl::c_all_of(streams_, [](const Stream& stream) {
        return stream.state == Stream::State::kDone;
      })) {
    mode_ = Mode::kDone;
    TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(SnapshotDoneFilePath(path_),
                                                   std::string(), env_));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> SnapshotManager::OwnerWorkerAddress(
    const std::string& stream_directory) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string worker_address;
  TF_RETURN_IF_ERROR(env_->FileExists(StreamWorkerFilePath(stream_directory)));
  TF_RETURN_IF_ERROR(ReadFileToString(
      env_, StreamWorkerFilePath(stream_directory), &worker_address));
  return worker_address;
}

absl::Status SnapshotManager::ReadOnDiskStream(
    int64_t stream_index, const std::string& worker_address,
    absl::flat_hash_set<int64_t>& global_split_indices)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  auto [it, success] = assignments_.insert({worker_address, stream_index});
  if (!success) {
    return absl::InternalError(absl::StrCat(
        "tf.data dispatcher failed to assign stream ", stream_index,
        " to snapshot worker ", worker_address,
        ": The  worker is already assigned stream ", it->second, "."));
  }

  std::string splits_path = SplitsDirectory(path_, stream_index);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> source_directories,
                      GetChildren(splits_path, env_));

  for (const auto& source_directory : source_directories) {
    std::string source_path = io::JoinPath(splits_path, source_directory);

    // `source_directory` must have this format: "source_<source_index>".
    std::vector<std::string> tokens = absl::StrSplit(source_directory, '_');
    int64_t source_index;
    if (tokens.size() != 2 || !absl::SimpleAtoi(tokens[1], &source_index) ||
        source_index < 0) {
      return absl::InternalError(absl::StrCat(
          "Can't parse the name of ", source_path,
          ": filename must have the format source_<source_index>"));
    }
    if (source_index >= num_sources()) {
      return absl::InternalError(
          absl::StrCat("Found conflict between the number of sources, ",
                       num_sources(), ", and the filename of ", source_path));
    }
    TF_RETURN_IF_ERROR(
        ReadOnDiskSource(stream_index, source_index, global_split_indices));
  }

  if (env_->FileExists(StreamDoneFilePath(path_, stream_index)).ok()) {
    streams_[stream_index].state = Stream::State::kDone;
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(bool assignment_added,
                      assignment_manager_.TryAddAssignment(
                          path_, worker_address, stream_index));
  if (!assignment_added) {
    return absl::InternalError(
        absl::StrCat("Failed to recover tf.data snapshot dispatcher: Worker ",
                     worker_address, " was assigned too many streams. At most ",
                     assignment_manager_.worker_max_concurrent_snapshots(),
                     " streams are allowed."));
  }
  return OkStatus();
}

namespace {

// Used for sorting repetition directories, based on the repetition index.
bool IsPriorRepetition(std::string repetition_dir1,
                       std::string repetition_dir2) {
  if (IsTemporaryFile(repetition_dir1)) {
    repetition_dir1 = *ParseTemporaryFile(repetition_dir1);
  }
  if (IsTemporaryFile(repetition_dir2)) {
    repetition_dir2 = *ParseTemporaryFile(repetition_dir2);
  }
  absl::StatusOr<int64_t> repetition_index1 =
      ParseRepetitionDirectoryName(repetition_dir1);
  absl::StatusOr<int64_t> repetition_index2 =
      ParseRepetitionDirectoryName(repetition_dir2);
  if (!repetition_index1.ok() || !repetition_index2.ok()) {
    return false;
  }
  return *repetition_index1 < *repetition_index2;
}

// Used for sorting split files, based on the local split index.
bool IsPriorSplit(std::string split1, std::string split2) {
  if (IsTemporaryFile(split1)) {
    split1 = *ParseTemporaryFile(split1);
  }
  if (IsTemporaryFile(split2)) {
    split2 = *ParseTemporaryFile(split2);
  }
  auto split_index1 = ParseSplitFilename(split1);
  auto split_index2 = ParseSplitFilename(split2);
  if (!split_index1.ok() || !split_index2.ok()) {
    return false;
  }
  return split_index1->first < split_index2->first;
}

}  // namespace

absl::Status SnapshotManager::ReadOnDiskSource(
    int64_t stream_index, int64_t source_index,
    absl::flat_hash_set<int64_t>& global_split_indices)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string source_directory =
      SourceDirectory(path_, stream_index, source_index);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> repetition_directories,
                      GetChildren(source_directory, env_));
  std::sort(repetition_directories.begin(), repetition_directories.end(),
            IsPriorRepetition);
  sources_[source_index].repetition_index =
      repetition_directories.empty() ? 0 : repetition_directories.size() - 1;

  for (const std::string& repetition : repetition_directories) {
    std::string repetition_dir =
        tsl::io::JoinPath(source_directory, repetition);
    std::vector<std::string> split_files;
    TF_RETURN_IF_ERROR(env_->GetChildren(repetition_dir, &split_files));
    std::sort(split_files.begin(), split_files.end(), IsPriorSplit);
    for (const std::string& split_file : split_files) {
      std::string split_path = io::JoinPath(repetition_dir, split_file);
      TF_RETURN_IF_ERROR(ReadOnDiskSplit(source_index, split_files, split_path,
                                         global_split_indices));
    }
    streams_[stream_index].num_assigned_splits_per_source[source_index] +=
        split_files.size();
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::ReadOnDiskSplit(
    int64_t source_index, const std::vector<std::string>& split_files,
    std::string split_file, absl::flat_hash_set<int64_t>& global_split_indices)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (IsTemporaryFile(split_file)) {
    TF_RETURN_IF_ERROR(
        RecoverSplit(split_file, *sources_[source_index].split_provider));
    TF_ASSIGN_OR_RETURN(split_file, ParseTemporaryFile(split_file));
  } else {
    // To account for this split having been assigned, skip a split in the
    // respective split provider.
    TF_RETURN_IF_ERROR(
        GetNextSplit(*sources_[source_index].split_provider).status());
  }

  // `split_file` must have this format:
  // "split_<local_split_index>_<global_split_index>".
  TF_ASSIGN_OR_RETURN(auto split_indices, ParseSplitFilename(split_file));
  auto [local_split_index, global_split_index] = split_indices;
  if (global_split_indices.contains(global_split_index)) {
    return absl::InternalError(absl::StrCat(
        "Found duplicate global split index in name of ", split_file));
  }
  global_split_indices.insert(global_split_index);
  return absl::OkStatus();
}

absl::Status SnapshotManager::RecoverSplit(const std::string& temp_split_file,
                                           SplitProvider& split_provider) {
  TF_ASSIGN_OR_RETURN(std::string recovered_split_file,
                      ParseTemporaryFile(temp_split_file));
  TF_ASSIGN_OR_RETURN(Tensor split, GetNextSplit(split_provider));
  // Uses the same temp file for split recovery. If the dispatcher fails during
  // recovery, there will be at most one temporary files for the same split.
  return AtomicallyWriteTFRecords(recovered_split_file, {split},
                                  kSplitFileCompression, temp_split_file, env_);
}

absl::StatusOr<Tensor> SnapshotManager::GetNextSplit(
    SplitProvider& split_provider) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider.GetNext(&split, &end_of_splits));
  while (end_of_splits) {
    TF_RETURN_IF_ERROR(split_provider.Reset());
    TF_RETURN_IF_ERROR(split_provider.GetNext(&split, &end_of_splits));
  }
  return split;
}

absl::Status SnapshotManager::HandleStreamCompletion(
    int64_t stream_index, absl::string_view worker_address)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  streams_[stream_index].state = Stream::State::kDone;
  assignment_manager_.RemoveAssignment(path_, worker_address, stream_index);
  ++num_completed_streams_;
  if (absl::c_all_of(streams_, [](const Stream& stream) {
        return stream.state == Stream::State::kDone;
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
  int64_t new_stream_index = streams_.size();
  TF_ASSIGN_OR_RETURN(bool assignment_added,
                      assignment_manager_.TryAddAssignment(
                          path_, worker_address, new_stream_index));
  if (!assignment_added) {
    return std::optional<int64_t>();
  }
  for (int64_t source_index = 0; source_index < num_sources(); ++source_index) {
    for (int64_t repetition_index = 0;
         repetition_index <= sources_[source_index].repetition_index;
         ++repetition_index) {
      TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(RepetitionDirectory(
          path_, new_stream_index, source_index, repetition_index)));
    }
  }
  TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(
      StreamWorkerFilePath(path_, new_stream_index), worker_address, env_));
  streams_.push_back(Stream(num_sources()));
  assignments_[worker_address] = new_stream_index;
  LOG(INFO) << "For snapshot at " << path_ << ", created stream_"
            << new_stream_index << " and assigned to " << worker_address;
  return new_stream_index;
}

absl::StatusOr<std::optional<int64_t>>
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
      return std::optional<int64_t>();
    }
    if (snapshot_progress->status().code() != error::OK) {
      TF_RETURN_IF_ERROR(
          HandleStreamError(worker_address, snapshot_progress->status()));
      return std::optional<int64_t>();
    }
  }
  if (!assigned_stream_index) {
    if (mode_ != Mode::kActive) {
      return std::optional<int64_t>();
    }
    TF_ASSIGN_OR_RETURN(assigned_stream_index,
                        MaybeCreateAndAssignNewStream(worker_address));
  }
  if (assigned_stream_index &&
      streams_[*assigned_stream_index].state == Stream::State::kDone) {
    return std::optional<int64_t>();
  }
  return assigned_stream_index;
}

absl::Status SnapshotManager::WorkerHeartbeat(
    const WorkerHeartbeatRequest& request, WorkerHeartbeatResponse& response)
    TF_LOCKS_EXCLUDED(mu_) {
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
  TF_ASSIGN_OR_RETURN(std::optional<int64_t> assigned_stream_index,
                      MaybeGetOrCreateStreamAssignment(request.worker_address(),
                                                       snapshot_progress));
  if (!assigned_stream_index) {
    return absl::OkStatus();
  }

  SnapshotTaskDef* snapshot_task = response.add_snapshot_tasks();
  snapshot_task->set_base_path(path_);
  snapshot_task->set_num_sources(num_sources());
  *snapshot_task->mutable_metadata() = metadata_;
  snapshot_task->set_stream_index(*assigned_stream_index);
  return absl::OkStatus();
}

absl::Status SnapshotManager::GetSnapshotSplit(
    const GetSnapshotSplitRequest& request, GetSnapshotSplitResponse& response)
    TF_LOCKS_EXCLUDED(mu_) {
  Tensor split;
  bool end_of_splits = false;
  int64_t local_split_index = 0;
  int64_t global_split_index = 0;
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

    Stream& stream = streams_[request.stream_index()];
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

    TF_RETURN_IF_ERROR(source.split_provider->GetNext(&split, &end_of_splits));
    if (end_of_splits) {
      response.set_end_of_splits(true);
      return absl::OkStatus();
    }

    ++stream.num_assigned_splits_per_source[request.source_index()];
    ++num_assigned_splits_;
  }
  std::string split_path = SplitPath(
      path_, request.stream_index(), request.source_index(),
      request.repetition_index(), local_split_index, global_split_index);
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(split_path, {split},
                                              kSplitFileCompression, env_));
  split.AsProtoTensorContent(response.mutable_split());
  return absl::OkStatus();
}

absl::Status SnapshotManager::ResetSource(Source& source, int64_t source_index)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(source.split_provider->Reset());
  ++source.repetition_index;
  LOG(INFO) << "Starting repetition_" << source.repetition_index << " "
            << "for snapshot " << path_ << ", source " << source_index;
  for (int64_t i = 0; i < streams_.size(); ++i) {
    TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(RepetitionDirectory(
        path_, /*stream_index=*/i, source_index, source.repetition_index)));
  }
  return absl::OkStatus();
}

absl::Status SnapshotManager::GetSnapshotStreams(
    GetSnapshotStreamsResponse& response) TF_LOCKS_EXCLUDED(mu_) {
  tsl::tf_shared_lock l(mu_);
  for (int64_t i = 0; i < streams_.size(); ++i) {
    SnapshotStreamInfo* stream = response.add_streams();
    stream->set_index(i);
    stream->set_state(streams_[i].state == Stream::State::kDone
                          ? SnapshotStreamInfo::DONE
                          : SnapshotStreamInfo::ASSIGNED);
  }
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
