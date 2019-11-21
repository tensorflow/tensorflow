/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/device_state.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

DeviceState::DeviceState(se::StreamExecutor* executor,
                         bool synchronous_deallocation, bool asynchronous,
                         bool allow_event_reuse)
    : synchronous_deallocation_(synchronous_deallocation),
      event_pool_(allow_event_reuse),
      compute_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
  compute_stream_ = absl::make_unique<se::Stream>(executor);
  host_to_device_stream_ = absl::make_unique<se::Stream>(executor);
  device_to_host_stream_ = absl::make_unique<se::Stream>(executor);
  callback_stream_ = absl::make_unique<se::Stream>(executor);
  compute_stream_->Init();
  host_to_device_stream_->Init();
  device_to_host_stream_->Init();
  callback_stream_->Init();
  device_to_device_streams_.reserve(kNumDeviceToDeviceStreams);
  for (int i = 0; i < kNumDeviceToDeviceStreams; ++i) {
    auto stream = absl::make_unique<se::Stream>(executor);
    stream->Init();
    device_to_device_streams_.push_back(std::move(stream));
  }
  execute_thread_ = absl::make_unique<WorkerThread>(tensorflow::Env::Default(),
                                                    "py_xla_execute");
  callback_thread_ = absl::make_unique<WorkerThread>(tensorflow::Env::Default(),
                                                     "py_xla_callback");
}

DeviceState::~DeviceState() {
  Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }
}

Status DeviceState::SynchronizeAllActivity() {
  Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  status.Update(compute_stream_->BlockHostUntilDone());
  status.Update(callback_stream_->BlockHostUntilDone());
  bool ok = compute_stream_->parent()->SynchronizeAllActivity();
  if (!ok) {
    status.Update(Unknown("SynchronizeAllActivity failed."));
  }
  return status;
}

Status DeviceState::ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                             se::Stream* dst_stream,
                                             se::DeviceMemoryBase src_buffer,
                                             se::DeviceMemoryBase dst_buffer) {
  // The default implementation simply calls ThenMemcpyD2D, and assumes that
  // the buffer addresses identify the devices. This does not work
  // on all platforms; this method is virtual so it can be overridden.
  transfer_stream->ThenMemcpyD2D(&dst_buffer, src_buffer, dst_buffer.size());
  return Status::OK();
}

void DeviceState::ThenExecuteOnCallbackThread(
    se::Stream* stream, std::function<void()> callback) const {
  stream->ThenDoHostCallback([this, callback]() mutable {
    callback_thread_->Schedule(std::move(callback));
  });
}

se::Stream* DeviceState::GetDeviceToDeviceStream() {
  absl::MutexLock lock(&mu_);
  int i = next_device_to_device_stream_;
  next_device_to_device_stream_ =
      (next_device_to_device_stream_ + 1) % device_to_device_streams_.size();
  return device_to_device_streams_.at(i).get();
}

}  // namespace xla
