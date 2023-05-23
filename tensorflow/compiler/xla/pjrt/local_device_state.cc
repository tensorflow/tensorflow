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

#include "tensorflow/compiler/xla/pjrt/local_device_state.h"

#include <memory>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace xla {

LocalDeviceState::LocalDeviceState(se::StreamExecutor* executor,
                                   LocalClient* client,
                                   AllocationModel allocation_model,
                                   int max_inflight_computations,
                                   bool allow_event_reuse,
                                   bool use_callback_stream)
    : allocation_model_(allocation_model),
      event_pool_(allow_event_reuse),
      compute_semaphore_(
          /*capacity=*/max_inflight_computations),
      executor_(executor),
      client_(client),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()) {
  compute_stream_ = std::make_unique<se::Stream>(executor);
  host_to_device_stream_ = std::make_unique<se::Stream>(executor);
  compute_stream_->Init();
  host_to_device_stream_->Init();
  if (use_callback_stream) {
    callback_stream_map_ =
        absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>();
  }
  device_to_host_streams_.reserve(kNumDeviceToHostStreams);
  for (int i = 0; i < kNumDeviceToHostStreams; ++i) {
    auto stream = std::make_unique<se::Stream>(executor);
    stream->Init();
    device_to_host_streams_.push_back(std::move(stream));
  }
  device_to_device_streams_.reserve(kNumDeviceToDeviceStreams);
  for (int i = 0; i < kNumDeviceToDeviceStreams; ++i) {
    auto stream = std::make_unique<se::Stream>(executor);
    stream->Init();
    device_to_device_streams_.push_back(std::move(stream));
  }
  execute_thread_ =
      std::make_unique<WorkerThread>(tsl::Env::Default(), "py_xla_execute");
  callback_thread_ =
      std::make_unique<WorkerThread>(tsl::Env::Default(), "py_xla_callback");
}

LocalDeviceState::~LocalDeviceState() {
  Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }
}

Status LocalDeviceState::SynchronizeAllActivity() {
  Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  status.Update(compute_stream_->BlockHostUntilDone());
  if (callback_stream_map_.has_value()) {
    for (auto& callback_stream : callback_stream_map_.value()) {
      status.Update(callback_stream.second->BlockHostUntilDone());
    }
  }
  for (auto& stream : device_to_host_streams_) {
    status.Update(stream->BlockHostUntilDone());
  }
  bool ok = compute_stream_->parent()->SynchronizeAllActivity();
  if (!ok) {
    status.Update(Unknown("SynchronizeAllActivity failed."));
  }
  return status;
}

Status LocalDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceMemoryBase src_buffer, se::DeviceMemoryBase dst_buffer) {
  // The default implementation simply calls ThenMemcpyD2D, and assumes that
  // the buffer addresses identify the devices. This does not work
  // on all platforms; this method is virtual so it can be overridden.
  transfer_stream->ThenMemcpyD2D(&dst_buffer, src_buffer, dst_buffer.size());
  return OkStatus();
}

void LocalDeviceState::ThenExecuteCallback(se::Stream* stream,
                                           std::function<void()> callback) {
  tsl::profiler::TraceMe traceme("ThenExecuteCallback");
  if (callback_stream_map_.has_value()) {
    // Prevent concurrent updates to the callback stream map.
    absl::MutexLock lock(&mu_);
    auto callback_stream = callback_stream_map_->find(stream);
    if (callback_stream == callback_stream_map_->end()) {
      auto new_stream = std::make_unique<se::Stream>(executor_);
      new_stream->Init();
      callback_stream =
          callback_stream_map_->insert({stream, std::move(new_stream)}).first;
    }
    callback_stream->second->ThenWaitFor(stream);
    stream = callback_stream->second.get();
  }
  stream->ThenDoHostCallback([this, callback{std::move(callback)}]() mutable {
    callback_thread_->Schedule(std::move(callback));
  });
}

se::Stream* LocalDeviceState::GetDeviceToHostStream() {
  absl::MutexLock lock(&mu_);
  int i = next_device_to_host_stream_;
  next_device_to_host_stream_ =
      (next_device_to_host_stream_ + 1) % device_to_host_streams_.size();
  return device_to_host_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetDeviceToDeviceStream() {
  absl::MutexLock lock(&mu_);
  int i = next_device_to_device_stream_;
  next_device_to_device_stream_ =
      (next_device_to_device_stream_ + 1) % device_to_device_streams_.size();
  return device_to_device_streams_.at(i).get();
}

std::unique_ptr<se::Stream> LocalDeviceState::BorrowStreamFromPool() {
  absl::MutexLock lock(&mu_);
  if (usage_stream_pool_.empty()) {
    auto stream = std::make_unique<se::Stream>(compute_stream_->parent());
    stream->Init();
    return stream;
  } else {
    std::unique_ptr<se::Stream> stream = std::move(usage_stream_pool_.top());
    usage_stream_pool_.pop();
    auto status = stream->RefreshStatus();  // Can return error::Unimplemented
    // Stream may fail with "ABORTED: Bad connection".
    if (status.code() != tsl::error::ABORTED) {
      CHECK(stream->ok()) << status;
    }
    return stream;
  }
}

void LocalDeviceState::ReturnStreamToPool(std::unique_ptr<se::Stream> stream) {
  auto status = stream->RefreshStatus();  // Can return error::Unimplemented
  // Stream may fail with "ABORTED: Bad connection".
  if (status.code() != tsl::error::ABORTED) {
    CHECK(stream->ok()) << status;
  }
  absl::MutexLock lock(&mu_);
  usage_stream_pool_.push(std::move(stream));
}

int LocalDeviceState::GetNewPrngSeed() {
  absl::MutexLock lock(&mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

}  // namespace xla
