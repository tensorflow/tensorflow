/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"

#include "tensorflow/compiler/xla/ptr_util.h"

namespace xla {
namespace gpu {

se::Stream* InfeedManager::GetStream(se::StreamExecutor* executor) {
  tensorflow::mutex_lock l(host_to_device_stream_mu_);
  if (host_to_device_executor_ == nullptr) {
    host_to_device_executor_ = executor;
    host_to_device_stream_ = MakeUnique<se::Stream>(executor);
    host_to_device_stream_->Init();
  }

  if (executor != host_to_device_executor_) {
    // The requested executor must be the same as the one for which
    // the stream is cached.
    return nullptr;
  }

  return host_to_device_stream_.get();
}

InfeedManager* GetOrCreateInfeedManager() {
  static InfeedManager* manager = new InfeedManager;
  return manager;
}

}  // namespace gpu
}  // namespace xla
