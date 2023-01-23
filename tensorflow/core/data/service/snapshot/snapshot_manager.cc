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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

using ::tsl::OkStatus;
using ::tsl::errors::InvalidArgument;

StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Start(
    const SnapshotRequest& request, Env* env) {
  SnapshotManager* snapshot_manager = new SnapshotManager(request.path(), env);
  TF_RETURN_IF_ERROR(snapshot_manager->Start(request));
  return absl::WrapUnique(snapshot_manager);
}

Status SnapshotManager::Start(const SnapshotRequest& request) {
  if (env_->FileExists(request.path()).ok()) {
    return InvalidArgument(request.path(), " already exists");
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
  TF_RETURN_IF_ERROR(
      WriteBinaryProto(env_, DatasetDefFilePath(path_), request.dataset()));
  return OkStatus();
}

StatusOr<std::unique_ptr<SnapshotManager>> SnapshotManager::Resume(
    absl::string_view path, Env* env) {
  SnapshotManager* snapshot_manager = new SnapshotManager(path, env);
  TF_RETURN_IF_ERROR(snapshot_manager->Resume());
  return absl::WrapUnique(snapshot_manager);
}

Status SnapshotManager::Resume() {
  if (!env_->FileExists(path_).ok()) {
    return InvalidArgument("failed to recover snapshot at ", path_,
                           ": the snapshot path doesn't exist");
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

  std::vector<std::string> stream_directories;
  TF_RETURN_IF_ERROR(env_->GetChildren(streams_path, &stream_directories));
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

  return OkStatus();
}

Status SnapshotManager::ReadOnDiskStream(
    int64_t stream_index, absl::flat_hash_set<int64_t>& global_split_indices) {
  std::string splits_path = SplitsDirectory(path_, stream_index);
  std::vector<std::string> source_directories;
  TF_RETURN_IF_ERROR(env_->GetChildren(splits_path, &source_directories));
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

  // TODO(mpcallanan): Handle unknowns.

  return OkStatus();
}

Status SnapshotManager::ReadOnDiskSource(
    int64_t stream_index, int64_t source_index,
    absl::flat_hash_set<int64_t>& global_split_indices) {
  std::string source_path = SourceDirectory(path_, stream_index, source_index);

  std::vector<std::string> split_filenames;
  TF_RETURN_IF_ERROR(env_->GetChildren(source_path, &split_filenames));

  Tensor unused_tensor;
  bool unused_end_of_splits;
  for (const auto& split_filename : split_filenames) {
    std::string split_path = io::JoinPath(source_path, split_filename);

    // `split_filename` must have this format:
    // "split_<local_split_index>_<global_split_index>".
    std::vector<std::string> tokens = absl::StrSplit(split_filename, '_');
    int64_t local_split_index;
    int64_t global_split_index;
    if (tokens.size() != 3 ||
        !absl::SimpleAtoi(tokens[1], &local_split_index) ||
        local_split_index < 0 ||
        !absl::SimpleAtoi(tokens[2], &global_split_index) ||
        global_split_index < 0) {
      return InvalidArgument("can't parse the name of ", split_path);
    }
    if (local_split_index > global_split_index) {
      return InvalidArgument(
          "found conflict between local split index and global split index in ",
          "name of ", split_path);
    }
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

StatusOr<int64_t> SnapshotManager::CreateNewStream(
    const std::string& worker_address) {
  int64_t new_stream_index = streams_.size();

  for (int64_t source_index = 0; source_index < num_sources(); ++source_index) {
    TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(
        SourceDirectory(path_, new_stream_index, source_index)));
  }

  streams_.push_back(Stream(num_sources()));
  assignments_.insert({worker_address, new_stream_index});
  VLOG(1) << "creating stream " << new_stream_index
          << " and assigning it to worker " << worker_address;

  return new_stream_index;
}

Status SnapshotManager::WorkerHeartbeat(const WorkerHeartbeatRequest& request,
                                        WorkerHeartbeatResponse& response) {
  SnapshotTaskDef* snapshot_task = response.add_snapshot_tasks();
  snapshot_task->set_base_path(path_);
  snapshot_task->set_num_sources(num_sources());
  *snapshot_task->mutable_metadata() = metadata_;

  if (auto it = assignments_.find(request.worker_address());
      it != assignments_.end()) {
    snapshot_task->set_stream_index(it->second);
    return OkStatus();
  }

  // TODO(mpcallanan): Handle orphans.

  TF_ASSIGN_OR_RETURN(int64_t new_stream_index,
                      CreateNewStream(request.worker_address()));
  snapshot_task->set_stream_index(new_stream_index);
  return OkStatus();
}

Status SnapshotManager::GetSnapshotSplit(const GetSnapshotSplitRequest& request,
                                         GetSnapshotSplitResponse& response) {
  // TODO(mpcallanan): Validate the request.

  Tensor split;
  bool end_of_splits;
  TF_RETURN_IF_ERROR(split_providers_[request.source_index()]->GetNext(
      &split, &end_of_splits));

  Stream& stream = streams_[request.stream_index()];
  if (end_of_splits) {
    // TODO(mpcallanan): Handle doneness.
    response.set_end_of_splits(true);
    return OkStatus();
  }

  std::string split_path = SplitPath(
      path_, request.stream_index(), request.source_index(),
      stream.num_assigned_splits[request.source_index()], num_assigned_splits_);
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecord(split_path, split, env_));

  ++stream.num_assigned_splits[request.source_index()];
  ++num_assigned_splits_;

  split.AsProtoTensorContent(response.mutable_split());

  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
