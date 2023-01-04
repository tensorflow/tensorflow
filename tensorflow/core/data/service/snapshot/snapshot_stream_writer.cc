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
#include "tensorflow/core/data/service/snapshot/snapshot_stream_writer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {

constexpr int64_t SnapshotWriterParams::kDefaultMaxChunkSizeBytes;

SnapshotStreamWriter::SnapshotStreamWriter(
    const SnapshotWriterParams& params, std::unique_ptr<TaskIterator> iterator)
    : params_(params),
      iterator_(std::move(iterator)),
      snapshot_thread_(RunSnapshotThread()) {}

Status SnapshotStreamWriter::Wait() TF_LOCKS_EXCLUDED(mu_) {
  snapshot_thread_.reset();
  mutex_lock l(mu_);
  return status_;
}

std::unique_ptr<Thread> SnapshotStreamWriter::RunSnapshotThread() {
  auto snapshot_fn = [this]() TF_LOCKS_EXCLUDED(mu_) {
    Status status = WriteSnapshotFn();
    if (!status.ok()) {
      mutex_lock l(mu_);
      status_ = std::move(status);
    }
  };
  return absl::WrapUnique(params_.env->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_snapshot_thread",
      std::move(snapshot_fn)));
}

Status SnapshotStreamWriter::WriteSnapshotFn() TF_LOCKS_EXCLUDED(mu_) {
  TF_RETURN_IF_ERROR(CreateChunksDirectory());
  while (ShouldWriteChunk()) {
    TF_RETURN_IF_ERROR(WriteChunk());
  }
  mutex_lock l(mu_);
  return status_;
}

Status SnapshotStreamWriter::CreateChunksDirectory() {
  return params_.env->RecursivelyCreateDir(
      UncommittedChunksDirectory(params_.snapshot_path, params_.stream_id));
}

bool SnapshotStreamWriter::ShouldWriteChunk() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::WriteChunk() {
  std::string chunk_file_path = GetChunkFilePath();
  snapshot_util::TFRecordWriter writer(chunk_file_path, params_.compression);
  TF_RETURN_IF_ERROR(writer.Initialize(params_.env));
  auto cleanup = gtl::MakeCleanup([&writer] { writer.Close().IgnoreError(); });

  while (ShouldWriteRecord()) {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  }
  return CommitChunk(chunk_file_path);
}

std::string SnapshotStreamWriter::GetChunkFilePath() const {
  return tsl::io::JoinPath(
      UncommittedChunksDirectory(params_.snapshot_path, params_.stream_id),
      absl::StrCat("chunk_", chunk_index_));
}

Status SnapshotStreamWriter::CommitChunk(const std::string& chunk_file_path) {
  // TODO(b/258691666): Write checkpoints.
  std::string chunk_basename(tsl::io::Basename(chunk_file_path));
  std::string committed_chunk_filename = tsl::io::JoinPath(
      CommittedChunksDirectory(params_.snapshot_path), chunk_basename);
  TF_RETURN_IF_ERROR(
      params_.env->RenameFile(chunk_file_path, committed_chunk_filename));
  ++chunk_index_;
  chunk_size_bytes_ = 0;
  return OkStatus();
}

bool SnapshotStreamWriter::ShouldWriteRecord() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return chunk_size_bytes_ < params_.max_chunk_size_bytes &&
         !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::WriteRecord(
    snapshot_util::TFRecordWriter& writer) {
  std::vector<Tensor> element;
  TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_sequence_));
  if (end_of_sequence_) {
    return writer.Close();
  }
  TF_RETURN_IF_ERROR(writer.WriteTensors(element));
  chunk_size_bytes_ += EstimatedSizeBytes(element);
  return OkStatus();
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  status_ = errors::Cancelled(
      "The tf.data service snapshot writer has been cancelled.");
}

}  // namespace data
}  // namespace tensorflow
