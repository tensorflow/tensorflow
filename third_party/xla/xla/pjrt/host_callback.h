/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PJRT_HOST_CALLBACK_H_
#define XLA_PJRT_HOST_CALLBACK_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/ffi/api/ffi.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "tsl/platform/logging.h"

// The following provides an API for implementing host callbacks on top of
// PjRT's send/recv interface (see xla::SendCallback and xla::RecvCallback).
// While this is not the only way to implement host callbacks using send/recv,
// it is provided as an example implementation that encapsulates common
// mechanisms for host callbacks in a framework-agnostic manner.

namespace xla {

bool ThisThreadIsInsideHostCallback();

void EnterHostCallback();

void LeaveHostCallback();

// A thread-safe queue for passing PjRtChunk objects for e.g. from Send ops to
// Recv ops.
class ThreadSafePjRtChunkQueue {
 public:
  // Push a PjRtChunk into the queue.
  void Push(PjRtChunk chunk) {
    absl::MutexLock lock(&mu_);
    if (promises_.empty()) {
      queue_.push_back(std::move(chunk));
      return;
    }
    auto pop_promise = promises_.front();
    pop_promise.Set(std::move(chunk));
    promises_.pop_front();
  }

  // Pop a PjRtChunk future from the queue.
  PjRtFuture<PjRtChunk> Pop() {
    absl::MutexLock lock(&mu_);
    if (queue_.empty()) {
      auto promise = PjRtFuture<PjRtChunk>::CreatePromise();
      promises_.push_back(promise);
      return PjRtFuture<PjRtChunk>(std::move(promise));
    }

    auto chunk = PjRtFuture<PjRtChunk>(std::move(queue_.front()));
    queue_.pop_front();
    return chunk;
  }

 private:
  absl::Mutex mu_;
  std::deque<PjRtChunk> queue_ ABSL_GUARDED_BY(mu_);
  // Contains unfulfilled pop promises.
  std::deque<PjRtFuture<PjRtChunk>::Promise> promises_ ABSL_GUARDED_BY(mu_);
};

struct HostCallbackArgInfo {
  // The channel_id associated with this value in HLO.
  uint16_t channel_id;
  // The host shape for this value.
  Shape shape;
};

struct HostCallback {
  // The metadata (e.g. channel_id, shape) for the operands and results.
  std::vector<HostCallbackArgInfo> operands;
  std::vector<HostCallbackArgInfo> results;

  // The host callback function takes two pointer arrays, each element of which
  // points to allocated host buffer according to corresponding operand or
  // result's shape. The first is for the outputs and the second is for the
  // inputs. The buffers are only guaranteed to be alive during the call. The
  // callback can also return error status to indicate the entire execution
  // should fail.
  std::function<absl::Status(void**, void**)> callback;
};

// A helper class that maintains the send/recv states for a host callback.
class HostCallbackContext {
 public:
  HostCallbackContext(
      HostCallback host_callback,
      bool use_major_to_minor_data_layout_for_callbacks,
      PjRtHostMemoryForDeviceManager* host_memory_for_device_manager)
      : host_callback_(std::move(host_callback)),
        use_major_to_minor_data_layout_for_callbacks_(
            use_major_to_minor_data_layout_for_callbacks),
        host_memory_for_device_manager_(host_memory_for_device_manager),
        args_(host_callback_.operands.size()),
        result_channels_(host_callback_.results.size()),
        ready_count_(args_.size()) {
    if (!use_major_to_minor_data_layout_for_callbacks_) {
      CHECK(host_memory_for_device_manager_);
    }
    for (auto& channel : result_channels_) {
      channel = std::make_unique<ThreadSafePjRtChunkQueue>();
    }
  }

  absl::Status OnSend(int arg_num, const PjRtTransferMetadata& metadata,
                      PjRtChunk data);

  void Receive(int res_num, const PjRtTransferMetadata& metadata,
               std::unique_ptr<CopyToDeviceStream> stream);

  const HostCallback& host_callback() const { return host_callback_; }

 private:
  HostCallback host_callback_;
  bool use_major_to_minor_data_layout_for_callbacks_;
  PjRtHostMemoryForDeviceManager* host_memory_for_device_manager_ = nullptr;
  std::vector<PjRtChunk> args_;
  std::vector<std::unique_ptr<ThreadSafePjRtChunkQueue>> result_channels_;
  std::atomic<int> ready_count_;
};

// The execution states for host callbacks for all replicas. The states are kept
// as vectors of vectors. The outer vector corresponds to the execution
// replicas. The inner vector is a list of host callback states for a single
// execution replica.
struct HostCallbackStates {
  std::vector<std::vector<std::unique_ptr<HostCallbackContext>>> contexts;
  std::vector<std::vector<SendCallback>> send_callbacks;
  std::vector<std::vector<RecvCallback>> recv_callbacks;
};

// Creates the execution context for the `host_callback` for one
// replica.
//
// `use_major_to_minor_data_layout_for_callbacks` should match the value set in
// the corresponding ExecuteOptions; see the comment there for more
// info. `host_memory_for_device_manager` may be nullptr if
// `use_major_to_minor_data_layout_for_callbacks` is true.
std::unique_ptr<HostCallbackContext>
CreateHostCallbackStateAndAppendSendRecvCallbacks(
    HostCallback host_callback,
    PjRtHostMemoryForDeviceManager* host_memory_for_device_manager,
    std::vector<SendCallback>& send_callbacks,
    std::vector<RecvCallback>& recv_callbacks,
    bool use_major_to_minor_data_layout_for_callbacks);

struct FfiLoadedHostCallbacks {
  static ffi::TypeId id;
  void** callbacks;
  uint32_t num_callbacks;
};

}  // namespace xla

#endif  // XLA_PJRT_HOST_CALLBACK_H_
