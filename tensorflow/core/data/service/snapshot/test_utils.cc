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
#include "tensorflow/core/data/service/snapshot/test_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/standalone.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

absl::StatusOr<std::string> CreateTmpDirectory() {
  std::string snapshot_path;
  if (!Env::Default()->LocalTempFilename(&snapshot_path)) {
    return absl::FailedPreconditionError(
        "Failed to create local temp file for snapshot.");
  }
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(snapshot_path)));
  return snapshot_path;
}

absl::StatusOr<int64_t> CommittedChunkIndex(const std::string& chunk_file) {
  std::vector<std::string> tokens = absl::StrSplit(chunk_file, '_');
  int64_t result = 0;
  if (tokens.size() != 4 || !absl::SimpleAtoi(tokens[2], &result)) {
    return absl::InternalError("Invalid");
  }
  return result;
}

absl::StatusOr<int64_t> CheckpointIndex(const std::string& checkpoint_file) {
  std::vector<std::string> tokens = absl::StrSplit(checkpoint_file, '_');
  int64_t result = 0;
  if (tokens.size() != 3 || !absl::SimpleAtoi(tokens[1], &result)) {
    return absl::InternalError("Invalid");
  }
  return result;
}

}  // namespace

PartialSnapshotWriter::PartialSnapshotWriter(const DatasetDef& dataset,
                                             const std::string& snapshot_path,
                                             int64_t stream_index,
                                             const std::string& compression,
                                             ByteSize max_chunk_size,
                                             absl::Duration checkpoint_interval)
    : dataset_(dataset),
      snapshot_path_(snapshot_path),
      stream_index_(stream_index),
      compression_(compression),
      max_chunk_size_(max_chunk_size),
      checkpoint_interval_(checkpoint_interval) {}

absl::StatusOr<PartialSnapshotWriter> PartialSnapshotWriter::Create(
    const DatasetDef& dataset, const std::string& snapshot_path,
    int64_t stream_index, const std::string& compression,
    ByteSize max_chunk_size, absl::Duration checkpoint_interval) {
  PartialSnapshotWriter writer(dataset, snapshot_path, stream_index,
                               compression, max_chunk_size,
                               checkpoint_interval);
  TF_RETURN_IF_ERROR(writer.Initialize());
  return writer;
}

absl::Status PartialSnapshotWriter::Initialize() {
  TF_ASSIGN_OR_RETURN(tmp_snapshot_path_, CreateTmpDirectory());
  // Each chunk contains one record.
  SnapshotWriterParams writer_params{tmp_snapshot_path_,
                                     stream_index_,
                                     compression_,
                                     Env::Default(),
                                     max_chunk_size_,
                                     checkpoint_interval_,
                                     /*test_only_keep_temp_files=*/true};
  TF_ASSIGN_OR_RETURN(std::unique_ptr<StandaloneTaskIterator> iterator,
                      TestIterator(dataset_));
  SnapshotStreamWriter snapshot_writer(writer_params, std::move(iterator));
  return snapshot_writer.Wait().status();
}

absl::Status PartialSnapshotWriter::WriteCommittedChunks(
    const absl::flat_hash_set<int64_t>& committed_chunk_indexes) const {
  std::string tmp_chunks_directory =
      CommittedChunksDirectory(tmp_snapshot_path_);
  std::string committed_chunks_directory =
      CommittedChunksDirectory(snapshot_path_);
  TF_RETURN_IF_ERROR(
      Env::Default()->RecursivelyCreateDir(committed_chunks_directory));
  std::vector<std::string> tmp_chunks;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetChildren(tmp_chunks_directory, &tmp_chunks));

  for (const std::string& tmp_chunk : tmp_chunks) {
    TF_ASSIGN_OR_RETURN(int64_t chunk_index, CommittedChunkIndex(tmp_chunk));
    if (committed_chunk_indexes.contains(chunk_index)) {
      std::string tmp_chunk_path =
          tsl::io::JoinPath(tmp_chunks_directory, tmp_chunk);
      std::string committed_chunk_path =
          tsl::io::JoinPath(committed_chunks_directory, tmp_chunk);
      TF_RETURN_IF_ERROR(
          Env::Default()->CopyFile(tmp_chunk_path, committed_chunk_path));
    }
  }
  return absl::OkStatus();
}

absl::Status PartialSnapshotWriter::WriteUncommittedChunks(
    const absl::flat_hash_set<int64_t>& uncommitted_chunk_indexes) const {
  std::string tmp_chunks_directory =
      CommittedChunksDirectory(tmp_snapshot_path_);
  std::string uncommitted_chunks_directory =
      UncommittedChunksDirectory(snapshot_path_, stream_index_);
  TF_RETURN_IF_ERROR(
      Env::Default()->RecursivelyCreateDir(uncommitted_chunks_directory));
  std::vector<std::string> tmp_chunks;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetChildren(tmp_chunks_directory, &tmp_chunks));

  for (const std::string& tmp_chunk : tmp_chunks) {
    TF_ASSIGN_OR_RETURN(int64_t chunk_index, CommittedChunkIndex(tmp_chunk));
    if (uncommitted_chunk_indexes.contains(chunk_index)) {
      std::string tmp_chunk_path =
          tsl::io::JoinPath(tmp_chunks_directory, tmp_chunk);
      std::string uncommitted_chunk_path = tsl::io::JoinPath(
          uncommitted_chunks_directory,
          absl::StrCat("chunk_", chunk_index, "_CHUNK_SHARDS_0"));
      TF_RETURN_IF_ERROR(
          Env::Default()->CopyFile(tmp_chunk_path, uncommitted_chunk_path));
    }
  }
  return absl::OkStatus();
}

absl::Status PartialSnapshotWriter::WriteCheckpoints(
    const absl::flat_hash_set<int64_t>& checkpoint_indexes) const {
  std::string tmp_checkpoints_directory =
      CheckpointsDirectory(tmp_snapshot_path_, stream_index_);
  std::string checkpoints_directory =
      CheckpointsDirectory(snapshot_path_, stream_index_);
  TF_RETURN_IF_ERROR(
      Env::Default()->RecursivelyCreateDir(checkpoints_directory));
  std::vector<std::string> tmp_checkpoints;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetChildren(tmp_checkpoints_directory, &tmp_checkpoints));

  for (const std::string& tmp_checkpoint : tmp_checkpoints) {
    TF_ASSIGN_OR_RETURN(int64_t checkpoint_index,
                        CheckpointIndex(tmp_checkpoint));
    if (checkpoint_indexes.contains(checkpoint_index)) {
      std::string tmp_checkpoint_path =
          tsl::io::JoinPath(tmp_checkpoints_directory, tmp_checkpoint);
      std::string checkpoint_path =
          tsl::io::JoinPath(checkpoints_directory, tmp_checkpoint);
      TF_RETURN_IF_ERROR(
          Env::Default()->CopyFile(tmp_checkpoint_path, checkpoint_path));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<StandaloneTaskIterator>> TestIterator(
    const DatasetDef& dataset_def) {
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_RETURN_IF_ERROR(dataset->MakeIterator(&iterator));
  return std::make_unique<StandaloneTaskIterator>(std::move(dataset),
                                                  std::move(iterator));
}

}  // namespace testing
}  // namespace data
}  // namespace tensorflow
