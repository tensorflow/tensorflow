/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/parallel_tfrecord_writer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/data/service/snapshot/utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {

ParallelTFRecordWriter::ParallelTFRecordWriter(const std::string& file_prefix,
                                               const std::string& compression,
                                               tsl::Env* env,
                                               ByteSize max_file_size,
                                               int64_t num_write_threads,
                                               int64_t buffer_size)
    : env_(env),
      file_prefix_(file_prefix),
      compression_(compression),
      max_file_size_(max_file_size),
      buffer_size_(buffer_size) {
  thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      env_, tsl::ThreadOptions{}, "write_tfrecord_thread", num_write_threads);
  for (int64_t i = 0; i < num_write_threads; ++i) {
    thread_pool_->Schedule([this]() { WriteFiles(); });
  }
}

ParallelTFRecordWriter::~ParallelTFRecordWriter() {
  absl::Status status = Finalize().status();
  if (!status.ok()) {
    LOG(ERROR) << "Parallel TFRecord writer failed with error: " << status;
  }
}

absl::Status ParallelTFRecordWriter::Write(std::vector<Tensor> record)
    ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(mu_);
  while (status_.ok() && !finalized_ && buffer_.size() >= buffer_size_) {
    ready_to_push_.Wait(&mu_);
  }
  TF_RETURN_IF_ERROR(status_);
  if (finalized_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Trying to write a closed TFRecord file at ", file_prefix_, "."));
  }

  buffer_.push_back(std::move(record));
  ready_to_pop_.Signal();
  return absl::OkStatus();
}

absl::StatusOr<ParallelTFRecordWriter::FileToStatsMap>
ParallelTFRecordWriter::Finalize() ABSL_LOCKS_EXCLUDED(mu_) {
  {
    absl::MutexLock l(mu_);
    finalized_ = true;
    ready_to_push_.SignalAll();
    ready_to_pop_.SignalAll();
  }

  thread_pool_.reset();
  absl::MutexLock l(mu_);
  TF_RETURN_IF_ERROR(status_);
  return file_stats_;
}

void ParallelTFRecordWriter::WriteFiles() {
  while (HasNext()) {
    UpdateStatus(WriteFile());
  }
}

bool ParallelTFRecordWriter::HasNext() const ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(mu_);
  if (!status_.ok()) {
    return false;
  }
  return !finalized_ || !buffer_.empty();
}

absl::Status ParallelTFRecordWriter::WriteFile() ABSL_LOCKS_EXCLUDED(mu_) {
  TF_ASSIGN_OR_RETURN(const std::string filename, GetUniqueFile());
  snapshot_util::TFRecordWriter writer(filename, compression_);
  TF_RETURN_IF_ERROR(writer.Initialize(env_));
  while (ShouldWriteFile(filename)) {
    TF_RETURN_IF_ERROR(WriteRecord(filename, writer));
  }
  TF_RETURN_IF_ERROR(writer.Close());
  return DeleteEmptyFile(filename);
}

bool ParallelTFRecordWriter::ShouldWriteFile(const std::string& filename) const
    ABSL_LOCKS_EXCLUDED(mu_) {
  if (!HasNext()) {
    return false;
  }
  absl::MutexLock l(mu_);
  auto iterator = file_stats_.find(filename);
  return iterator == file_stats_.end() ||
         iterator->second.estimated_size < max_file_size_;
}

absl::Status ParallelTFRecordWriter::WriteRecord(
    const std::string& filename, snapshot_util::TFRecordWriter& writer) {
  TF_ASSIGN_OR_RETURN(std::optional<std::vector<Tensor>> record,
                      GetNextRecord(filename));
  if (!record.has_value()) {
    return absl::OkStatus();
  }

  tsl::profiler::TraceMe activity("WriteTFRecord",
                                  tsl::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(writer.WriteTensors(*std::move(record)));
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::vector<Tensor>>>
ParallelTFRecordWriter::GetNextRecord(const std::string& filename)
    ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(mu_);
  while (status_.ok() && !finalized_ && buffer_.empty()) {
    ready_to_pop_.Wait(&mu_);
  }
  TF_RETURN_IF_ERROR(status_);
  if (buffer_.empty()) {
    return std::nullopt;
  }

  std::vector<Tensor> record = std::move(buffer_.front());
  ByteSize estimated_size = EstimatedSize(record);
  LOG_EVERY_N_SEC(INFO, 1) << "Writing TFRecord of " << estimated_size
                           << " to file " << filename << "*.";
  ++file_stats_[filename].num_records;
  file_stats_[filename].estimated_size += estimated_size;
  buffer_.pop_front();
  ready_to_push_.SignalAll();
  return record;
}

absl::Status ParallelTFRecordWriter::DeleteEmptyFile(
    const std::string& filename) ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(mu_);
  auto iterator = file_stats_.find(filename);
  if (iterator != file_stats_.end() && iterator->second.num_records > 0) {
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(env_->DeleteFile(filename));
  if (iterator != file_stats_.end()) {
    file_stats_.erase(iterator);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ParallelTFRecordWriter::GetUniqueFile() const {
  std::string filename = absl::StrCat(file_prefix_, "__shard__",
                                      absl::Hex(tsl::random::New64()), "_");
  if (!env_->CreateUniqueFileName(&filename, ".tfrecord")) {
    return absl::InternalError(
        absl::StrCat("Failed to write file ", filename,
                     ": Unable to open temporary files."));
  }
  return filename;
}

void ParallelTFRecordWriter::UpdateStatus(absl::Status status)
    ABSL_LOCKS_EXCLUDED(mu_) {
  if (status.ok()) {
    return;
  }
  absl::MutexLock l(mu_);
  status_.Update(std::move(status));
  ready_to_push_.SignalAll();
  ready_to_pop_.SignalAll();
}
}  // namespace data
}  // namespace tensorflow
