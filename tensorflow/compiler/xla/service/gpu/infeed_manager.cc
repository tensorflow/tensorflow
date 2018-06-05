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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

InfeedManager::InfeedManager() : host_to_device_executor_(nullptr) {}

void InfeedManager::Reset() {
  tensorflow::mutex_lock l(mu_);
  CHECK(dequeued_buffer_.empty());
  for (auto buffer : enqueued_buffer_) {
    buffer->Done();
  }
  enqueued_buffer_.clear();
}

void InfeedManager::EnqueueBuffers(const std::vector<InfeedBuffer*>& buffers) {
  tensorflow::mutex_lock l(mu_);
  bool was_empty = enqueued_buffer_.empty();
  for (gpu::InfeedBuffer* b : buffers) {
    enqueued_buffer_.push_back(b);
  }
  if (was_empty) {
    // This has the potential to suffer from the notified thread
    // immediately trying and failing to acquire mu_, but seems
    // preferable to the alternative of notifying outside the lock
    // on every enqueue.
    cv_.notify_one();
  }
}

InfeedBuffer* InfeedManager::BlockingDequeueBuffer() {
  bool became_empty = false;
  InfeedBuffer* current_buffer;
  {
    tensorflow::mutex_lock l(mu_);
    while (enqueued_buffer_.empty()) {
      cv_.wait(l);
    }
    current_buffer = enqueued_buffer_.front();
    enqueued_buffer_.pop_front();
    dequeued_buffer_.insert(current_buffer);
    if (enqueued_buffer_.empty()) {
      became_empty = true;
    }
  }
  if (became_empty) {
    for (const auto& callback : on_empty_callbacks_) {
      callback();
    }
  }
  return current_buffer;
}

void InfeedManager::ReleaseBuffers(const std::vector<InfeedBuffer*>& buffers) {
  {
    tensorflow::mutex_lock l(mu_);
    for (gpu::InfeedBuffer* b : buffers) {
      CHECK(ContainsKey(dequeued_buffer_, b));
      dequeued_buffer_.erase(b);
    }
  }
  for (gpu::InfeedBuffer* b : buffers) {
    b->Done();
  }
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

void InfeedManager::RegisterOnEmptyCallback(std::function<void()> callback) {
  on_empty_callbacks_.push_back(std::move(callback));
}

InfeedManager* GetOrCreateInfeedManager() {
  static InfeedManager* manager = new InfeedManager;
  return manager;
}

}  // namespace gpu
}  // namespace xla
