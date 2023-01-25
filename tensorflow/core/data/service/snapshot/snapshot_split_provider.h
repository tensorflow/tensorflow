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

namespace tensorflow {
namespace data {

// Split provider that supports writing distributed snapshots.
class SnapshotSplitProvider : public SplitProvider {
 public:
  SnapshotSplitProvider(const std::string& address, const std::string& protocol,
                        const SnapshotTaskDef& snapshot_task,
                        int64_t source_index, absl::Duration timeout);

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const std::string address_;
  const std::string protocol_;
  const SnapshotTaskDef snapshot_task_;
  const int64_t source_index_;
  const absl::Duration timeout_;

  mutex mu_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_SNAPSHOT_SPLIT_PROVIDER_H_
