/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/local_device_state.h"

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/protobuf/error_codes.pb.h"

namespace xla {

LocalDeviceState::LocalDeviceState(se::StreamExecutor* executor,
                                   LocalClient* client,
                                   AllocationModel allocation_model,
                                   int max_inflight_computations,
                                   bool allow_event_reuse,
                                   bool use_callback_stream, int device_ordinal,
                                   std::optional<StreamOptions> stream_options)
    : allocation_model_(allocation_model),
      event_pool_(allow_event_reuse),
      compute_semaphore_(
          /*capacity=*/max_inflight_computations),
      executor_(executor),
      client_(client),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()) {
  local_hardware_id_ = executor_->device_ordinal();
  local_device_id_ =
      device_ordinal != -1 ? device_ordinal : executor_->device_ordinal();

  int num_device_to_host_streams =
      stream_options.has_value() ? stream_options->num_device_to_host_streams
                                 : kNumDeviceToHostStreams;
  int num_device_to_device_streams =
      stream_options.has_value() ? stream_options->num_device_to_device_streams
                                 : kNumDeviceToDeviceStreams;
  auto create_stream = [executor, &stream_options]() {
    if (stream_options.has_value()) {
      return executor->CreateStream(stream_options->priority).value();
    } else {
      return executor->CreateStream().value();
    }
  };
  compute_stream_ = create_stream();
  host_to_device_stream_ = create_stream();
  if (use_callback_stream) {
    callback_stream_map_ =
        absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>();
  }
  device_to_host_streams_.reserve(num_device_to_host_streams);
  for (int i = 0; i < num_device_to_host_streams; ++i) {
    device_to_host_streams_.emplace_back(create_stream());
  }
  device_to_device_streams_.reserve(num_device_to_device_streams);
  for (int i = 0; i < num_device_to_device_streams; ++i) {
    device_to_device_streams_.emplace_back(create_stream());
  }
  fixed_size_pool_usage_streams_.reserve(kNumFixedSizePoolUsageStreams);
  for (int i = 0; i < kNumFixedSizePoolUsageStreams; ++i) {
    fixed_size_pool_usage_streams_.emplace_back(create_stream());
  }
  external_ready_event_streams_.reserve(kNumExternalReadyEventStreams);
  for (int i = 0; i < kNumExternalReadyEventStreams; ++i) {
    external_ready_event_streams_.emplace_back(create_stream());
  }
  execute_thread_ =
      std::make_unique<WorkerThread>(tsl::Env::Default(), "py_xla_execute");
  callback_thread_ =
      std::make_unique<WorkerThread>(tsl::Env::Default(), "py_xla_callback");
}

LocalDeviceState::~LocalDeviceState() {
  absl::Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }
}

absl::Status LocalDeviceState::SynchronizeAllActivity() {
  absl::Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  status.Update(compute_stream_->BlockHostUntilDone());
  if (callback_stream_map_.has_value()) {
    absl::MutexLock lock(&callback_stream_map_mu_);
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

absl::Status LocalDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceMemoryBase src_buffer, se::DeviceMemoryBase dst_buffer) {
  // The default implementation simply calls MemcpyD2D, and assumes that
  // the buffer addresses identify the devices. This does not work
  // on all platforms; this method is virtual so it can be overridden.
  return transfer_stream->MemcpyD2D(&dst_buffer, src_buffer, dst_buffer.size());
}

absl::Status LocalDeviceState::ThenExecuteCallback(
    se::Stream* stream, std::function<void()> callback) {
  tsl::profiler::TraceMe traceme("ThenExecuteCallback");
  if (callback_stream_map_.has_value()) {
    // Prevent concurrent updates to the callback stream map.
    absl::MutexLock lock(&callback_stream_map_mu_);
    auto callback_stream = callback_stream_map_->find(stream);
    if (callback_stream == callback_stream_map_->end()) {
      TF_ASSIGN_OR_RETURN(auto new_stream, executor_->CreateStream());
      callback_stream =
          callback_stream_map_->insert({stream, std::move(new_stream)}).first;
    }
    TF_RETURN_IF_ERROR(callback_stream->second->WaitFor(stream));
    stream = callback_stream->second.get();
  }
  return stream->DoHostCallback(
      [this, callback{std::move(callback)}]() mutable {
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

se::Stream* LocalDeviceState::GetFixedSizePoolUsageStream() {
  absl::MutexLock lock(&mu_);
  int i = next_fixed_size_pool_usage_stream_;
  next_fixed_size_pool_usage_stream_ =
      (next_fixed_size_pool_usage_stream_ + 1) %
      fixed_size_pool_usage_streams_.size();
  return fixed_size_pool_usage_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetExternalReadyEventStream() {
  absl::MutexLock lock(&mu_);
  int i = next_external_ready_event_stream_;
  next_external_ready_event_stream_ = (next_external_ready_event_stream_ + 1) %
                                      external_ready_event_streams_.size();
  return external_ready_event_streams_.at(i).get();
}

absl::StatusOr<se::Stream*> LocalDeviceState::GetStreamFromExternalStream(
    std::intptr_t stream) {
  // TODO(skyewm): replace with map lookup if performance is an issue (currently
  // it just iterates over 4 streams).
  for (const std::unique_ptr<se::Stream>& se_stream :
       external_ready_event_streams_) {
    if (absl::bit_cast<std::intptr_t>(
            se_stream->platform_specific_handle().stream) == stream) {
      return se_stream.get();
    }
  }
  return NotFound(
      "GetStreamFromExternalStream failed to find stream. Only GPU streams "
      "used for dlpack imports are supported.");
}

std::vector<se::Stream*> LocalDeviceState::GetDeviceToDeviceStreams() {
  absl::MutexLock lock(&mu_);
  std::vector<se::Stream*> result;
  result.reserve(device_to_device_streams_.size());
  for (const auto& stream : device_to_device_streams_) {
    result.push_back(stream.get());
  }
  return result;
}

std::unique_ptr<se::Stream> LocalDeviceState::BorrowStreamFromPool() {
  {
    absl::MutexLock lock(&stream_pool_mu_);
    if (!usage_stream_pool_.empty()) {
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

  // The stream pool is empty, create a new stream.
  auto stream = compute_stream_->parent()->CreateStream().value();
  return stream;
}

void LocalDeviceState::ReturnStreamToPool(std::unique_ptr<se::Stream> stream) {
  auto status = stream->RefreshStatus();  // Can return error::Unimplemented
  // Stream may fail with "ABORTED: Bad connection".
  if (status.code() != tsl::error::ABORTED) {
    CHECK(stream->ok()) << status;
  }
  absl::MutexLock lock(&stream_pool_mu_);
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
