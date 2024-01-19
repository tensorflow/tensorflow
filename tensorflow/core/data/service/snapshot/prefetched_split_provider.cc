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
#include "tensorflow/core/data/service/snapshot/prefetched_split_provider.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/lib/io/compression.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace data {

PrefetchedSplitProvider::PrefetchedSplitProvider(
    std::unique_ptr<SplitProvider> split_provider, const std::string& directory,
    tsl::Env* env, size_t num_write_threads, size_t buffer_size_per_thread)
    : env_(env),
      directory_(directory),
      num_write_threads_(num_write_threads),
      buffer_size_(num_write_threads_ * buffer_size_per_thread),
      split_provider_(std::move(split_provider)) {
  thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      env_, tsl::ThreadOptions{}, "tf_data_prefetch_splits_thread",
      num_write_threads_);
  for (size_t i = 0; i < num_write_threads_; ++i) {
    thread_pool_->Schedule([this]() { PrefetchLoop(); });
  }
}

absl::StatusOr<std::optional<Tensor>> PrefetchedSplitProvider::GetSplit(
    const std::string& target_split_path) ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(&mu_);
  while (status_.ok() && finished_threads_ < num_write_threads_ &&
         buffer_.empty()) {
    ready_to_pop_.Wait(&mu_);
  }
  TF_RETURN_IF_ERROR(status_);
  if (buffer_.empty()) {
    return std::nullopt;
  }

  SplitFile split_file = std::move(buffer_.front());
  TF_RETURN_IF_ERROR(env_->RenameFile(split_file.filename, target_split_path));
  buffer_.pop_front();
  ready_to_push_.Signal();
  return std::move(split_file.split);
}

void PrefetchedSplitProvider::PrefetchLoop() ABSL_LOCKS_EXCLUDED(mu_) {
  while (ShouldPrefetchSplit()) {
    absl::StatusOr<bool> has_next = PrefetchSplit();
    if (!has_next.status().ok()) {
      UpdateStatus(has_next.status());
      break;
    }
    if (!*has_next) {
      break;
    }
  }

  absl::MutexLock l(&mu_);
  if (++finished_threads_ >= num_write_threads_) {
    ready_to_pop_.SignalAll();
  }
}

bool PrefetchedSplitProvider::ShouldPrefetchSplit() const
    ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(&mu_);
  return status_.ok();
}

absl::StatusOr<bool> PrefetchedSplitProvider::PrefetchSplit()
    ABSL_LOCKS_EXCLUDED(mu_) {
  TF_ASSIGN_OR_RETURN(std::optional<Tensor> split, GetSplitFromProvider());
  if (!split.has_value()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(std::string split_path, GetUniqueFile());
  TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
      split_path, {*split}, tsl::io::compression::kNone, env_));

  absl::MutexLock l(&mu_);
  buffer_.push_back({*std::move(split), std::move(split_path)});
  ready_to_pop_.Signal();
  return true;
}

absl::StatusOr<std::optional<Tensor>>
PrefetchedSplitProvider::GetSplitFromProvider() ABSL_LOCKS_EXCLUDED(mu_) {
  absl::MutexLock l(&mu_);
  while (status_.ok() && buffer_.size() >= buffer_size_) {
    ready_to_push_.Wait(&mu_);
  }
  TF_RETURN_IF_ERROR(status_);

  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, &end_of_splits));
  if (end_of_splits) {
    return std::nullopt;
  }
  return split;
}

absl::StatusOr<std::string> PrefetchedSplitProvider::GetUniqueFile() const {
  std::string filename = tsl::io::JoinPath(directory_, "split_");
  if (!env_->CreateUniqueFileName(&filename, ".tfrecord")) {
    return absl::InternalError(
        absl::StrCat("Failed to prefetch tf.data service split to ", filename,
                     ": Unable to open temporary file."));
  }
  return filename;
}

void PrefetchedSplitProvider::UpdateStatus(absl::Status status)
    ABSL_LOCKS_EXCLUDED(mu_) {
  if (status.ok()) {
    return;
  }
  absl::MutexLock l(&mu_);
  status_.Update(std::move(status));
  ready_to_push_.SignalAll();
  ready_to_pop_.SignalAll();
}
}  // namespace data
}  // namespace tensorflow
