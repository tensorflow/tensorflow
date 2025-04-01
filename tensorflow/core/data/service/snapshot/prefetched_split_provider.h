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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PREFETCHED_SPLIT_PROVIDER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PREFETCHED_SPLIT_PROVIDER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Uses multiple threads to prefetch splits and write them to temporary files.
// Used to speed up tf.data snapshot manager where splits should be persisted
// before returning to the users. This class is thread-safe.
//
// Usage example:
//
// std::unique_ptr<SplitProvider> split_provider = ...
// PrefetchedSplitProvider prefetched_split_provider(
//     std::move(split_provider), "/tmp/directory", Env::Default());
// TF_ASSIGN_OR_RETURN(std::optional<Tensor> split,
//                     prefetched_split_provider.GetSplit(SplitPath(...)));
// if (split.has_value) {
//   return *split;
// }
class PrefetchedSplitProvider {
 public:
  // Creates a prefetched split provider by prefetching given `split_provider`.
  // `directory` is where to write temporary splits. The splits will be moved to
  // a target file when returned to the client (see the comment for `GetSplit`).
  // `num_write_threads` is the number of threads to prefetch and write splits.
  // `buffer_size_per_thread` is the size of the buffer holding the prefetched
  // but unread splits. For every prefetched split, we keep: (1) an in-memory
  // Tensor in the buffer, and (2) an on-disk file representing the same split.
  explicit PrefetchedSplitProvider(
      std::unique_ptr<SplitProvider> split_provider,
      const std::string& directory, tsl::Env* env,
      size_t num_write_threads = 20, size_t buffer_size_per_thread = 5);
  virtual ~PrefetchedSplitProvider();
  PrefetchedSplitProvider(const PrefetchedSplitProvider&) = delete;
  PrefetchedSplitProvider& operator=(const PrefetchedSplitProvider&) = delete;

  // Writes the split to `target_split_path` and returns the split. Returns
  // `std::nullopt` if no more splits are available. If there are more available
  // splits but not currently ready for reading, blocks until they are ready.
  absl::StatusOr<std::optional<Tensor>> GetNext(const std::string& split_path);

  // Resets the split provider.
  absl::Status Reset();

  // Cancels the split provider. After cancelling, concurrent `GetNext` calls
  // will return a Cancelled error.
  void Cancel();

 private:
  // Prefetched split and its split index.
  struct SplitAndIndex {
    Tensor split;
    size_t index = 0;

    // Returns the absolute path of the prefetched split.
    std::string SplitPath(const std::string& directory) const {
      return tsl::io::JoinPath(directory,
                               absl::StrCat("split_", index, ".tfrecord"));
    }

    friend bool operator<(const SplitAndIndex& lhs, const SplitAndIndex& rhs) {
      return lhs.index < rhs.index;
    }
  };

  // Initializes directories for writing. This cleans up all existing files in
  // `directory_`.
  absl::Status InitDirs();

  // Runs the prefetch threads.
  std::unique_ptr<tsl::thread::ThreadPool> RunPrefetchThreads();

  // The prefetching threads run this method to prefetch the splits.
  void PrefetchLoop();

  // Whether the prefetching thread should try to fetch more splits.
  bool ShouldPrefetchSplit() const;

  // If there is enough buffer space, prefetches one split and writes it to a
  // temporary file. If the buffer is full, blocks until there is buffer space.
  absl::StatusOr<bool> PrefetchSplit();

  // Gets the next split from the split provider.
  absl::StatusOr<std::optional<SplitAndIndex>> GetSplitFromProvider();

  // Updates the status and notifies waiters.
  void UpdateStatus(absl::Status status);

  tsl::Env* const env_;
  const std::string directory_;
  const size_t num_write_threads_;
  const size_t buffer_size_;

  mutable absl::Mutex mu_;
  mutable absl::CondVar ready_to_push_;
  mutable absl::CondVar ready_to_pop_;

  std::unique_ptr<SplitProvider> split_provider_;

  absl::Status status_ ABSL_GUARDED_BY(mu_);

  // Whether the split provider is being reset.
  bool reset_ ABSL_GUARDED_BY(mu_) = false;

  // The indices ensure the splits are returned in order. When prefetching a
  // split, associates each split with the `split_index_to_write_`. The buffer
  // is sorted by the split index. When reading, waits for the split with index
  // `split_index_to_read_`.
  size_t split_index_to_read_ ABSL_GUARDED_BY(mu_) = 0;
  size_t split_index_to_write_ ABSL_GUARDED_BY(mu_) = 0;

  // Number of finished threads. If `finished_threads_ >= num_write_threads_`,
  // then all the splits have been pushed to the buffer. Otherwise, the split
  // provider has not produced all the splits, or some thread is still writing
  // splits to the files.
  size_t finished_threads_ ABSL_GUARDED_BY(mu_) = 0;

  // Buffer to hold the splits. The size should be bounded by `buffer_size_`.
  absl::btree_set<SplitAndIndex> buffer_ ABSL_GUARDED_BY(mu_);

  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_ ABSL_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_PREFETCHED_SPLIT_PROVIDER_H_
