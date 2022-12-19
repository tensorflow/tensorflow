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
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64_t kDefaultMaxChunkSizeBytes = 10 * (size_t{1} << 30);  // 10GB

constexpr const char kUncommittedChunksDirectory[] = "uncommitted_chunks";

}  // namespace

SnapshotStreamWriter::SnapshotStreamWriter(
    std::unique_ptr<TaskIterator> iterator,
    const std::string& snapshot_stream_path, Env* env,
    std::optional<int64_t> max_chunk_size_bytes)
    : env_(env),
      snapshot_stream_path_(snapshot_stream_path),
      max_chunk_size_bytes_(max_chunk_size_bytes.has_value()
                                ? *max_chunk_size_bytes
                                : kDefaultMaxChunkSizeBytes),
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
  return absl::WrapUnique(env_->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_snapshot_thread",
      std::move(snapshot_fn)));
}

Status SnapshotStreamWriter::WriteSnapshotFn() {
  TF_RETURN_IF_ERROR(CreateChunksDirectory());
  while (ShouldWriteChunk()) {
    TF_RETURN_IF_ERROR(WriteChunk());
  }
  return status();
}

Status SnapshotStreamWriter::CreateChunksDirectory() {
  return env_->RecursivelyCreateDir(
      absl::StrCat(snapshot_stream_path_, "/", kUncommittedChunksDirectory));
}

bool SnapshotStreamWriter::ShouldWriteChunk() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !end_of_sequence_ && status_.ok();
}

Status SnapshotStreamWriter::WriteChunk() {
  // TODO(b/258691666): Support compression.
  snapshot_util::TFRecordWriter writer(GetChunkPath(),
                                       tsl::io::compression::kNone);
  TF_RETURN_IF_ERROR(writer.Initialize(env_));
  auto cleanup = gtl::MakeCleanup([&writer] { writer.Close().IgnoreError(); });

  while (ShouldWriteRecord()) {
    TF_RETURN_IF_ERROR(WriteRecord(writer));
  }
  // TODO(b/258691666): Write checkpoints and move the chunks to the committed
  // chunks directory.
  InitializeNextChunk();
  return OkStatus();
}

std::string SnapshotStreamWriter::GetChunkPath() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return absl::StrCat(snapshot_stream_path_, "/", kUncommittedChunksDirectory,
                      "/chunk_", chunk_index_);
}

bool SnapshotStreamWriter::ShouldWriteRecord() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return chunk_size_bytes_ < max_chunk_size_bytes_ && !end_of_sequence_ &&
         status_.ok();
}

Status SnapshotStreamWriter::WriteRecord(snapshot_util::TFRecordWriter& writer)
    TF_LOCKS_EXCLUDED(mu_) {
  std::vector<Tensor> element;
  bool end_of_sequence = false;
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_sequence));
    end_of_sequence_ = end_of_sequence;
  }
  if (end_of_sequence) {
    return writer.Close();
  }
  TF_RETURN_IF_ERROR(writer.WriteTensors(element));
  mutex_lock l(mu_);
  chunk_size_bytes_ += EstimatedSizeBytes(element);
  return OkStatus();
}

void SnapshotStreamWriter::InitializeNextChunk() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  ++chunk_index_;
  chunk_size_bytes_ = 0;
}

void SnapshotStreamWriter::Cancel() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  status_ = errors::Cancelled(
      "The tf.data service snapshot writer has been cancelled.");
}

Status SnapshotStreamWriter::status() const TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return status_;
}

}  // namespace data
}  // namespace tensorflow
