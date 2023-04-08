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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_to_from_proto.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {

using ::tsl::OkStatus;
using ::tsl::errors::InvalidArgument;

// The time for which an UNKNOWN stream should transition to ORPHAN if no worker
// claims ownership of it via heartbeat.
const absl::Duration kUnknownStreamTimeout = absl::Seconds(45);

StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Start(
    const SnapshotRequest& request, Env* env) {
  SnapshotManager* snapshot_manager = new SnapshotManager(request.path(), env);
  TF_RETURN_IF_ERROR(snapshot_manager->Start(request));
  return absl::WrapUnique(snapshot_manager);
}

Status SnapshotManager::Start(const SnapshotRequest& request) {
  if (env_->FileExists(request.path()).ok()) {
    return InvalidArgument("Distributed tf.data snapshot at ", request.path(),
                           " already exists.");
  }
  TF_RETURN_IF_ERROR(CreateSplitProviders(request.dataset(), split_providers_));
  TF_RETURN_IF_ERROR(WriteOnDiskSkeleton());
  TF_RETURN_IF_ERROR(WriteOnDiskMetadata(request));
  metadata_ = request.metadata();
  return OkStatus();
}

Status SnapshotManager::WriteOnDiskSkeleton() {
  TF_RETURN_IF_ERROR(
      env_->RecursivelyCreateDir(CommittedChunksDirectory(path_)));
  TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(StreamsDirectory(path_)));
  return OkStatus();
}

Status SnapshotManager::WriteOnDiskMetadata(const SnapshotRequest& request) {
  TF_RETURN_IF_ERROR(WriteTextProto(env_, SnapshotMetadataFilePath(path_),
                                    request.metadata()));
  TF_RETURN_IF_ERROR(WriteStringToFile(env_, DatasetSpecFilePath(path_),
                                       request.metadata().element_spec()));
  TF_RETURN_IF_ERROR(
      WriteBinaryProto(env_, DatasetDefFilePath(path_), request.dataset()));
  return OkStatus();
}

StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Resume(
    absl::string_view path, Env* env) {
  SnapshotManager* snapshot_manager =
      new SnapshotManager(path, env, absl::Microseconds(env->NowMicros()));
  TF_RETURN_IF_ERROR(snapshot_manager->Resume());
  return absl::WrapUnique(snapshot_manager);
}

Status SnapshotManager::Resume() {
  if (!env_->FileExists(path_).ok()) {
    return InvalidArgument("failed to recover snapshot at ", path_,
                           ": the snapshot path doesn't exist");
  }
  if (env_->FileExists(SnapshotDoneFilePath(path_)).ok()) {
    mode_ = Mode::kDone;
    LOG(INFO) << "attempted to recover snapshot at " << path_
              << " but it's already done";
    return OkStatus();
  }
  if (env_->FileExists(SnapshotErrorFilePath(path_)).ok()) {
    mode_ = Mode::kError;
    StatusProto status_proto;
    TF_RETURN_IF_ERROR(
        ReadTextProto(env_, SnapshotErrorFilePath(path_), &status_proto));
    status_ = tsl::StatusFromProto(status_proto);
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(ReadOnDiskMetadata());
  TF_RETURN_IF_ERROR(ReadOnDiskStreams());
  return OkStatus();
}

Status SnapshotManager::ReadOnDiskMetadata() {
  if (!env_->FileExists(SnapshotMetadataFilePath(path_)).ok()) {
    return InvalidArgument("failed to recover snapshot at ", path_,
                           ": snapshot has no snapshot.metadata");
  }
  TF_RETURN_IF_ERROR(
      ReadTextProto(env_, SnapshotMetadataFilePath(path_), &metadata_));

  if (!env_->FileExists(DatasetDefFilePath(path_)).ok()) {
    return InvalidArgument("failed to recovery snapshot at ", path_,
                           ": snapshot has no dataset_def.proto");
  }
  DatasetDef dataset_def;
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(env_, DatasetDefFilePath(path_), &dataset_def));

  TF_RETURN_IF_ERROR(CreateSplitProviders(dataset_def, split_providers_));
  return OkStatus();
}

Status SnapshotManager::ReadOnDiskStreams() {
  std::string streams_path = StreamsDirectory(path_);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> stream_directories,
                      GetChildren(streams_path, env_));
  streams_.resize(stream_directories.size(), Stream(num_sources()));

  absl::flat_hash_set<int64_t> global_split_indices;
  for (const auto& stream_directory : stream_directories) {
    std::string stream_path = io::JoinPath(streams_path, stream_directory);

    // `stream_directory` must have this format: "stream_<stream_index>".
    std::vector<std::string> tokens = absl::StrSplit(stream_directory, '_');
    int64_t stream_index;
    if (tokens.size() != 2 || !absl::SimpleAtoi(tokens[1], &stream_index) ||
        stream_index < 0) {
      return InvalidArgument(
          "can't parse the name of ", stream_path,
          ": filename must have the format stream_<stream_index>");
    }

    TF_RETURN_IF_ERROR(ReadOnDiskStream(stream_index, global_split_indices));
  }

  for (int64_t i = 0; i < global_split_indices.size(); ++i) {
    if (!global_split_indices.contains(i)) {
      return InvalidArgument("found missing global split index, ", i, ", in ",
                             path_);
    }
  }
  num_assigned_splits_ = global_split_indices.size();

  if (!streams_.empty() &&
      std::all_of(streams_.begin(), streams_.end(),
                  [](const Stream& stream) { return stream.done; })) {
    mode_ = Mode::kDone;
    TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(SnapshotDoneFilePath(path_),
                                                   std::string(), env_));
  }

  return OkStatus();
}

Status SnapshotManager::ReadOnDiskStream(
    int64_t stream_index, absl::flat_hash_set<int64_t>& global_split_indices) {
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
      return InvalidArgument(
          "can't parse the name of ", source_path,
          ": filename must have the format source_<source_index>");
    }
    if (source_index >= num_sources()) {
      return InvalidArgument("found conflict between the number of sources, ",
                             num_sources(), ", and the filename of ",
                             source_path);
    }
    TF_RETURN_IF_ERROR(
        ReadOnDiskSource(stream_index, source_index, global_split_indices));
  }

  if (env_->FileExists(StreamDoneFilePath(path_, stream_index)).ok()) {
    streams_[stream_index].done = true;
    return OkStatus();
  }

  unknowns_.insert(stream_index);
  return OkStatus();
}

Status SnapshotManager::ReadOnDiskSource(
    int64_t stream_index, int64_t source_index,
    absl::flat_hash_set<int64_t>& global_split_indices) {
  std::string source_path = SourceDirectory(path_, stream_index, source_index);
  TF_ASSIGN_OR_RETURN(std::vector<std::string> split_filenames,
                      GetChildren(source_path, env_));

  Tensor unused_tensor;
  bool unused_end_of_splits;
  for (const auto& split_filename : split_filenames) {
    std::string split_path = io::JoinPath(source_path, split_filename);
    TF_ASSIGN_OR_RETURN(auto split_indices, ParseSplitFilename(split_filename));
    auto [local_split_index, global_split_index] = split_indices;
    if (local_split_index > split_filenames.size() - 1) {
      return InvalidArgument(
          "found conflict between the number of splits and name of ",
          split_path);
    }
    if (global_split_indices.contains(global_split_index)) {
      return InvalidArgument("found duplicate global split index in name of ",
                             split_path);
    }

    // To account for this split having been assigned, skip a split in the
    // respective provider.
    TF_RETURN_IF_ERROR(split_providers_[source_index]->GetNext(
        &unused_tensor, &unused_end_of_splits));
    global_split_indices.insert(global_split_index);
  }

  streams_[stream_index].num_assigned_splits[source_index] =
      split_filenames.size();
  return OkStatus();
}

Status SnapshotManager::HandleStreamCompletion(
    int64_t stream_index, absl::string_view worker_address) {
  streams_[stream_index].done = true;
  assignments_.erase(worker_address);
  if (assignments_.empty() && orphans_.empty() && unknowns_.empty()) {
    mode_ = Mode::kDone;
    TF_RETURN_IF_ERROR(AtomicallyWriteStringToFile(SnapshotDoneFilePath(path_),
                                                   std::string(), env_));
    LOG(INFO) << "Finished writing tf.data distributed snapshot at " << path_;
  }
  return OkStatus();
}

Status SnapshotManager::HandleStreamError(absl::string_view worker_address,
                                          const StatusProto& status_proto) {
  // This method returns an OkStatus as the RPC status if the worker reports an
  // error. The errors are communicated back to the workers with a proper RPC
  // response, instead of with a error status.
  if (!status_.ok()) {
    return OkStatus();
  }

  mode_ = Mode::kError;
  status_ = tsl::StatusFromProto(status_proto);
  TF_RETURN_IF_ERROR(AtomicallyWriteTextProto(SnapshotErrorFilePath(path_),
                                              status_proto, env_));
  LOG(ERROR) << "Failed to write tf.data distributed snapshot at " << path_
             << ". Worker " << worker_address << " reported error: " << status_;
  return OkStatus();
}

std::optional<int64_t> SnapshotManager::MaybeAssignOrphanStream(
    absl::string_view worker_address) {
  if (!orphans_.empty()) {
    int64_t stream_index = *orphans_.begin();
    orphans_.erase(orphans_.begin());
    assignments_[worker_address] = stream_index;
    LOG(INFO) << "assigning an existing stream, " << stream_index
              << ", to worker " << worker_address;
    return stream_index;
  }
  return std::nullopt;
}

StatusOr<int64_t> SnapshotManager::CreateAndAssignNewStream(
    absl::string_view worker_address) {
  int64_t new_stream_index = streams_.size();
  for (int64_t source_index = 0; source_index < num_sources(); ++source_index) {
    TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(
        SourceDirectory(path_, new_stream_index, source_index)));
  }
  streams_.push_back(Stream(num_sources()));
  assignments_[worker_address] = new_stream_index;
  LOG(INFO) << "assigning a new stream, " << new_stream_index << ", to worker "
            << worker_address;
  return new_stream_index;
}

void SnapshotManager::ReassignPreviouslyAssignedStream(
    int64_t stream_index, absl::string_view worker_address) {
  LOG(INFO) << "reassigning a previous assignment of stream " << stream_index
            << " to worker " << worker_address;
  assignments_[worker_address] = stream_index;
  orphans_.erase(stream_index);
  unknowns_.erase(stream_index);
}

StatusOr<std::optional<int64_t>>
SnapshotManager::MaybeGetOrCreateStreamAssignment(
    absl::string_view worker_address,
    const SnapshotTaskProgress* snapshot_progress) {
  std::optional<int64_t> assigned_stream_index;
  if (auto it = assignments_.find(worker_address); it != assignments_.end()) {
    assigned_stream_index = it->second;
  }
  if (snapshot_progress) {
    if (assigned_stream_index.has_value() &&
        *assigned_stream_index !=
            snapshot_progress->snapshot_task().stream_index()) {
      return errors::Internal("worker ", worker_address,
                              " think it's assigned stream ",
                              " but it's actually assigned assigned stream ",
                              *assigned_stream_index);
    }
    if (!assigned_stream_index &&
        stream_available(snapshot_progress->snapshot_task().stream_index())) {
      ReassignPreviouslyAssignedStream(
          snapshot_progress->snapshot_task().stream_index(), worker_address);
      assigned_stream_index = snapshot_progress->snapshot_task().stream_index();
    }
    if (assigned_stream_index.has_value() && snapshot_progress->completed()) {
      TF_RETURN_IF_ERROR(HandleStreamCompletion(
          snapshot_progress->snapshot_task().stream_index(), worker_address));
      assigned_stream_index.reset();
    }
    if (snapshot_progress->status().code() != error::OK) {
      TF_RETURN_IF_ERROR(
          HandleStreamError(worker_address, snapshot_progress->status()));
      return std::optional<int64_t>();
    }
  }
  if (!assigned_stream_index) {
    assigned_stream_index = MaybeAssignOrphanStream(worker_address);
  }
  if (!assigned_stream_index) {
    if (mode_ != Mode::kActive) {
      return std::optional<int64_t>();
    }
    TF_ASSIGN_OR_RETURN(assigned_stream_index,
                        CreateAndAssignNewStream(worker_address));
  }
  return assigned_stream_index;
}

Status SnapshotManager::WorkerHeartbeat(const WorkerHeartbeatRequest& request,
                                        WorkerHeartbeatResponse& response) {
  dead_workers_.erase(request.worker_address());

  if (mode_ == Mode::kDone || mode_ == Mode::kError) {
    // When the snapshot manager is done or in an error state, it returns an
    // empty response to inform the workers to cancel the ongoing tasks.
    return OkStatus();
  }

  const SnapshotTaskProgress* snapshot_progress = nullptr;
  if (auto it = request.snapshot_task_progress().find(path_);
      it != request.snapshot_task_progress().end()) {
    snapshot_progress = &it->second;
  }
  TF_ASSIGN_OR_RETURN(std::optional<int64_t> assigned_stream_index,
                      MaybeGetOrCreateStreamAssignment(request.worker_address(),
                                                       snapshot_progress));
  if (!assigned_stream_index) {
    return OkStatus();
  }

  SnapshotTaskDef* snapshot_task = response.add_snapshot_tasks();
  snapshot_task->set_base_path(path_);
  snapshot_task->set_num_sources(num_sources());
  *snapshot_task->mutable_metadata() = metadata_;
  snapshot_task->set_stream_index(*assigned_stream_index);
  return OkStatus();
}

Status SnapshotManager::GetSnapshotSplit(const GetSnapshotSplitRequest& request,
                                         GetSnapshotSplitResponse& response) {
  auto it = assignments_.find(request.worker_address());
  if (it == assignments_.end()) {
    if (!stream_available(request.stream_index()) ||
        dead_workers_.contains(request.worker_address())) {
      return StreamAssignmentChanged(request.worker_address(),
                                     request.stream_index());
    }
    ReassignPreviouslyAssignedStream(request.stream_index(),
                                     request.worker_address());
  } else if (it->second != request.stream_index()) {
    return errors::Internal("worker ", request.worker_address(),
                            " think it's assigned stream ",
                            request.stream_index(),
                            " but it's actually assigned stream ", it->second);
  }

  Tensor split;
  bool end_of_splits;
  TF_RETURN_IF_ERROR(split_providers_[request.source_index()]->GetNext(
      &split, &end_of_splits));

  Stream& stream = streams_[request.stream_index()];
  int64_t local_split_index =
      stream.num_assigned_splits[request.source_index()];
  int64_t global_split_index = num_assigned_splits_;
  response.set_local_split_index(local_split_index);
  if (end_of_splits) {
    if (mode_ == Mode::kActive) {
      mode_ = Mode::kWindingDown;
    }
    response.set_end_of_splits(true);
    return OkStatus();
  }

  std::string split_path =
      SplitPath(path_, request.stream_index(), request.source_index(),
                local_split_index, global_split_index);
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
      split_path, {split}, tsl::io::compression::kNone, env_));
  split.AsProtoTensorContent(response.mutable_split());

  ++stream.num_assigned_splits[request.source_index()];
  ++num_assigned_splits_;
  return OkStatus();
}

Status SnapshotManager::GetSnapshotStreams(
    GetSnapshotStreamsResponse& response) {
  for (int64_t i = 0; i < streams_.size(); ++i) {
    SnapshotStreamInfo* stream = response.add_streams();
    stream->set_index(i);
    if (orphans_.contains(i)) {
      stream->set_state(SnapshotStreamInfo::ORPHAN);
    } else if (unknowns_.contains(i)) {
      stream->set_state(SnapshotStreamInfo::UNKNOWN);
    } else {
      stream->set_state(streams_[i].done ? SnapshotStreamInfo::DONE
                                         : SnapshotStreamInfo::ASSIGNED);
    }
  }
  return OkStatus();
}

void SnapshotManager::HandleMissingWorker(const std::string& worker_address) {
  if (auto it = assignments_.find(worker_address); it != assignments_.end()) {
    LOG(INFO) << "deleting assignment for stream " << it->second
              << " due to lost worker " << worker_address;
    orphans_.insert(it->second);
    assignments_.erase(it);
    dead_workers_.insert(worker_address);
  }
}

void SnapshotManager::UpdateStreams() {
  // Check for streams to move from `unknowns_` to `orphans_`.
  if (resume_time_micros_.has_value() && !unknowns_.empty() &&
      absl::Microseconds(env_->NowMicros()) - resume_time_micros_.value() >
          kUnknownStreamTimeout) {
    for (auto stream_index : unknowns_) {
      orphans_.insert(stream_index);
    }
    unknowns_.clear();
  }
}

}  // namespace data
}  // namespace tensorflow
