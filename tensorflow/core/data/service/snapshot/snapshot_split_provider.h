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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_SPLIT_PROVIDER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_SPLIT_PROVIDER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Split provider that supports writing distributed snapshots.
class SnapshotSplitProvider : public SplitProvider {
 public:
  SnapshotSplitProvider(const std::string& worker_address,
                        const SnapshotTaskDef& snapshot_task,
                        int64_t source_index, absl::Duration timeout,
                        std::unique_ptr<DataServiceDispatcherClient> dispatcher,
                        Env* env);

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const std::string worker_address_;
  const SnapshotTaskDef snapshot_task_;
  const int64_t source_index_;
  const absl::Duration timeout_;
  Env* const env_;

  // Gets the next split from file or dispatcher and validates it.
  Status GetAndValidateSplit(Tensor* split, bool* end_of_splits);

  // Gets the next split by reading from the splits directory.
  Status GetSplitFromFile(const std::string& split_file, Tensor* split,
                          bool* end_of_splits);

  // Gets the next split by sending an RPC to the dispatcher. Returns the local
  // split index from the dispatcher.
  StatusOr<int64_t> GetSplitFromDispatcher(Tensor* split, bool* end_of_splits);

  // Reads from the split directory and returns a map of split index to absolute
  // file path of the split, starting at `start_index`.
  Status GetSplitsFiles(
      int64_t start_index,
      absl::btree_map<int64_t, std::string>& split_to_file_map,
      int64_t& repetition_index) const;

  // Verifies `split_files` contains consecutive splits starting at
  // `start_index`.
  Status ValidateSplitFiles(
      const absl::btree_map<int64_t, std::string>& split_files,
      int64_t start_index) const;

  // Verifies `split_files` contains consecutive splits starting at
  // `start_index` and ending at `end_index`.
  Status ValidateSplitFiles(
      const absl::btree_map<int64_t, std::string>& split_files,
      int64_t start_index, int64_t end_index, bool end_of_splits) const;

  mutable mutex mu_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_ TF_GUARDED_BY(mu_);

  // The next split to read.
  int64_t next_split_index_ TF_GUARDED_BY(mu_) = 0;

  // Number of times the dataset has repeated.
  int64_t repetition_index_ TF_GUARDED_BY(mu_) = 0;

  // Maps the local split index to the absolute split file path.
  absl::btree_map<int64_t, std::string> split_to_file_map_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_SPLIT_PROVIDER_H_
