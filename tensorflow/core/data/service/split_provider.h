/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_SPLIT_PROVIDER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SPLIT_PROVIDER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// SplitProvider which reads splits from a tf.data service dispatcher over RPC.
class DataServiceSplitProvider : public SplitProvider {
 public:
  DataServiceSplitProvider(const std::string& address,
                           const std::string& protocol, int64_t iteration_id,
                           int64_t split_provider_index, int64_t timeout_ms)
      : address_(address),
        protocol_(protocol),
        iteration_id_(iteration_id),
        split_provider_index_(split_provider_index),
        timeout_ms_(timeout_ms) {}

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const std::string address_;
  const std::string protocol_;
  const int64_t iteration_id_;
  const int64_t split_provider_index_;
  const int64_t timeout_ms_;

  mutex mu_;
  int64_t repetition_ TF_GUARDED_BY(mu_) = 0;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_ TF_GUARDED_BY(mu_);
};

// Makes split providers for `dataset_def` and stores them in `split_providers`.
Status CreateSplitProviders(
    const DatasetDef& dataset_def,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SPLIT_PROVIDER_H_
