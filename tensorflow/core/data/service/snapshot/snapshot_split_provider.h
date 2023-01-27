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
  SnapshotSplitProvider(const std::string& dispatcher_address,
                        const std::string& dispatcher_protocol,
                        const std::string& worker_address,
                        const SnapshotTaskDef& snapshot_task,
                        int64_t source_index, absl::Duration timeout, Env* env);

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const std::string dispatcher_address_;
  const std::string dispatcher_protocol_;
  const std::string worker_address_;
  const SnapshotTaskDef snapshot_task_;
  const int64_t source_index_;
  const absl::Duration timeout_;
  Env* const env_;

  // If the next split is written to the file system, returns the name of the
  // split file. If it is not written to the file system, returns NotFound.
  StatusOr<std::string> GetSplitFilename() const;

  // Gets the next split by reading from the splits directory.
  Status GetSplitFromFile(const std::string& split_file, Tensor* split,
                          bool* end_of_splits);

  // Gets the next split by sending an RPC to the dispatcher.
  Status GetSplitFromDispatcher(Tensor* split, bool* end_of_splits);

  mutable mutex mu_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_ TF_GUARDED_BY(mu_);
  int64_t next_split_index_ TF_GUARDED_BY(mu_) = 0;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_SPLIT_PROVIDER_H_
