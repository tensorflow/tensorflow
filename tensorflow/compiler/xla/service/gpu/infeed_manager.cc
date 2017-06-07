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

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

InfeedManager::InfeedManager()
    : current_buffer_(nullptr),
      host_to_device_executor_(nullptr) {}

void InfeedManager::Reset() {
  tensorflow::mutex_lock l(mu_);
  CHECK(!current_buffer_);
  for (auto buffer : enqueued_buffer_) {
    buffer->Done();
  }
  enqueued_buffer_.clear();
}

void InfeedManager::EnqueueBuffer(InfeedBuffer* buffer) {
  tensorflow::mutex_lock l(mu_);
  bool was_empty = enqueued_buffer_.empty();
  enqueued_buffer_.push_back(buffer);
  if (was_empty) {
    // This has the potential to suffer from the notified thread
    // immediately trying and failing to acquire mu_, but seems
    // preferable to the alternative of notifying outside the lock
    // on every enqueue.
    cv_.notify_one();
  }
}

InfeedBuffer* InfeedManager::BlockingDequeueBuffer() {
  tensorflow::mutex_lock l(mu_);
  while (enqueued_buffer_.empty()) {
    cv_.wait(l);
  }
  CHECK(!current_buffer_);
  current_buffer_ = enqueued_buffer_.front();
  enqueued_buffer_.pop_front();
  return current_buffer_;
}

void InfeedManager::ReleaseCurrentBuffer(se::DeviceMemoryBase* device_memory) {
  tensorflow::mutex_lock l(mu_);
  CHECK(current_buffer_);
  CHECK(device_memory->IsSameAs(*current_buffer_->device_memory()));
  current_buffer_->Done();
  current_buffer_ = nullptr;
}

se::Stream* InfeedManager::GetStream(se::StreamExecutor* executor) {
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
