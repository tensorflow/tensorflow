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

#include "tensorflow/core/data/service/split_provider.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

Status DataServiceSplitProvider::GetNext(Tensor* split, bool* end_of_splits)
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (!dispatcher_) {
    dispatcher_ =
        std::make_unique<DataServiceDispatcherClient>(address_, protocol_);
  }
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [this, split, end_of_splits]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return dispatcher_->GetSplit(iteration_id_, repetition_,
                                     split_provider_index_, *split,
                                     *end_of_splits);
      },
      "get next split",
      /*deadline_micros=*/Env::Default()->NowMicros() +
          (timeout_ms_ * EnvTime::kMillisToMicros)));
  if (*end_of_splits) {
    VLOG(1) << "Reached end of splits for iteration_id=" << iteration_id_
            << ", repetition=" << repetition_;
  } else {
    VLOG(1) << "Requested split: " << split->DebugString()
            << "; with iteration_id=" << iteration_id_
            << ", repetition=" << repetition_;
  }
  return absl::OkStatus();
}

Status DataServiceSplitProvider::Reset() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  repetition_++;
  return absl::OkStatus();
}

Status DataServiceSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  return errors::Unimplemented(
      "Save is not implemented for DataServiceSplitProvider");
}

Status DataServiceSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  return errors::Unimplemented(
      "Restore is not implemented for DataServiceSplitProvider");
}

Status CreateSplitProviders(
    const DatasetDef& dataset_def,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers) {
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> standalone_dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(params, dataset_def.graph(),
                                                    &standalone_dataset));
  TF_RETURN_IF_ERROR(standalone_dataset->MakeSplitProviders(&split_providers));
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
