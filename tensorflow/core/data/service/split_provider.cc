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

#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {

Status DataServiceSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
  mutex_lock l(mu_);
  if (!dispatcher_) {
    dispatcher_ =
        absl::make_unique<DataServiceDispatcherClient>(address_, protocol_);
  }
  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [this, split, end_of_splits] {
        return dispatcher_->GetSplit(iteration_id_, iteration_,
                                     split_provider_index_, *split,
                                     *end_of_splits);
      },
      "get next split",
      /*deadline_micros=*/Env::Default()->NowMicros() +
          (timeout_ms_ * EnvTime::kMillisToMicros)));
  if (*end_of_splits) {
    VLOG(1) << "Reached end of splits for iteration_id=" << iteration_id_
            << ", iteration=" << iteration_;
  } else {
    VLOG(1) << "Requested split: " << split->DebugString()
            << "; with iteration_id=" << iteration_id_
            << ", iteration=" << iteration_;
  }
  return Status::OK();
}

Status DataServiceSplitProvider::Reset() {
  mutex_lock l(mu_);
  iteration_++;
  return Status::OK();
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

}  // namespace data
}  // namespace tensorflow
