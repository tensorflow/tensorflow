/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLA_EXECUTOR_STATE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLA_EXECUTOR_STATE_H_

#include <memory>

#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"

// Defines XLA:GPU specific state that will be attached to the GpuExecutor.

namespace xla {
namespace gpu {

class GpuExecutorXLAState {
 public:
  explicit GpuExecutorXLAState(stream_executor::StreamExecutor *) {}

  InfeedManager *getOrCreateInfeedManager(stream_executor::StreamExecutor *se) {
    tensorflow::mutex_lock l(this->mu_);
    if (!infeed_manager_) infeed_manager_ = std::make_unique<InfeedManager>(se);
    return infeed_manager_.get();
  }

  OutfeedManager *getOrCreateOutfeedManager(
      stream_executor::StreamExecutor *se) {
    tensorflow::mutex_lock l(this->mu_);
    if (!outfeed_manager_)
      outfeed_manager_ = std::make_unique<OutfeedManager>();
    return outfeed_manager_.get();
  }

 private:
  template <typename T>
  T *getOrCreate(stream_executor::StreamExecutor *se, std::unique_ptr<T> &ptr) {
    if (!ptr) {
      ptr = std::make_unique<T>(se);
    }
    return ptr.get();
  }

  tensorflow::mutex mu_;
  std::unique_ptr<InfeedManager> infeed_manager_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<OutfeedManager> outfeed_manager_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLA_EXECUTOR_STATE_H_
