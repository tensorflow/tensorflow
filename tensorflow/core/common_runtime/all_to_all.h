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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_ALL_TO_ALL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_ALL_TO_ALL_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device.h"

namespace tensorflow {

// Implementation of collective all-to-all.
class AllToAll : public CollectiveImplementationInterface {
 public:
  AllToAll();

  void Run(StatusCallback done) override;

  Status InitializeCollectiveParams(CollectiveParams* col_params) override {
    return Status::OK();
  }

  // Initializes members of CollectiveContext not yet initialized, i.e. device
  // and device_locality.  Also saves the CollectiveContext in this object.
  Status InitializeCollectiveContext(
      std::shared_ptr<CollectiveContext> col_ctx) override;

 private:
  std::shared_ptr<CollectiveContext> col_ctx_;
  const CollectiveParams* col_params_;  // Not owned
  std::vector<Tensor> input_chunks_;
  Tensor output_buffer_;
  std::vector<Tensor> output_chunks_;
  StatusCallback done_;
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
  int counter_ TF_GUARDED_BY(mu_);

  void DispatchSend(int src_rank, int target_rank, const Tensor* tensor,
                    const StatusCallback& done);

  void DispatchRecv(int src_rank, int target_rank, Tensor* tensor,
                    const StatusCallback& done);

  // Atomically increments counter_ by one for sending, one for receiving.
  // Invokes done when counter_ reaches 2.
  // The purpose of checking counter_ is to ensure that done_ is called once.
  StatusCallback CheckCounterAndCallDone();
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_ALL_TO_ALL_H_
