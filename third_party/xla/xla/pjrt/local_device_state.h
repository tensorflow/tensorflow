/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PJRT_LOCAL_DEVICE_STATE_H_
#define XLA_PJRT_LOCAL_DEVICE_STATE_H_

#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stack>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// Class that encapsulates state relating to a device (e.g., a GPU) on which we
// can perform computation and transfers. LocalDeviceState objects only exist
// for devices local to this host.
class LocalDeviceState {
 public:
  // There are three different semantics used by memory allocators on different
  // devices.
  enum AllocationModel {
    // kSynchronous is used by CPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer cannot be freed until after the last stream operation
    // referencing the buffer has completed, so the client is responsible for
    // keeping buffers alive until all device-side activity that consumes those
    // buffers has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the last stream using a buffer.
    kSynchronous,

    // kComputeSynchronous is used by GPU devices.
    //
    // A buffer returned from the allocator at time t can be used after the
    // compute stream has finished executing the last computation enqueued
    // before time t.
    //
    // A buffer b can be freed after:
    //   1) The last use of b on the compute stream has been enqueued, and
    //   2) For any non-compute stream s on which an operation o using b is
    //      enqueued, either:
    //     a) The host has been notified that o has completed, or
    //     b) The next operation to be enqueued on the compute stream is
    //        guaranteed to be started after o has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the compute stream.
    kComputeSynchronized,

    // kAsynchronous is used by TPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer b can be freed as soon as the last stream operation using b has
    // been enqueued.
    //
    // The allocator and lower-level runtime are responsible for keeping buffers
    // alive (if that is needed) from the perspective of the device until any
    // device-side work actually completes.
    //
    // The only exception is when a buffer is transferred between devices since
    // only one of the device executors knows about the transfer, so the buffer
    // must be manually kept alive from the perspective of the other executor.
    kAsynchronous
  };

  // Options for stream creations.
  struct StreamOptions {
    int priority = 0;
    int num_device_to_host_streams = 1;
    int num_device_to_device_streams = 1;
  };

  // `device_ordinal` is the logical local device ordinal (returned by
  // `local_device_id()`), and it's used to look up an addressable device local
  // to a given client. If it is not set (-1 by default), the device's logical
  // device ordinal will be the same as its physical device ordinal (returned by
  // `local_hardware_id()`). In general, different PJRT devices have different
  // logical device ordinals, and several PJRT devices can have the same
  // physical device ordinal if they share the same physical device.
  LocalDeviceState(se::StreamExecutor* executor, LocalClient* client,
                   AllocationModel allocation_model,
                   int max_inflight_computations, bool allow_event_reuse,
                   bool use_callback_stream, int device_ordinal = -1,
                   std::optional<StreamOptions> stream_options = std::nullopt);
  virtual ~LocalDeviceState();

  se::StreamExecutor* executor() const { return executor_; }

  PjRtLocalDeviceId local_device_id() { return local_device_id_; }
  PjRtLocalHardwareId local_hardware_id() { return local_hardware_id_; }

  LocalClient* client() const { return client_; }

  AllocationModel allocation_model() const { return allocation_model_; }

  EventPool& event_pool() { return event_pool_; }

  se::Stream* compute_stream() const { return compute_stream_.get(); }
  se::Stream* host_to_device_stream() const {
    return host_to_device_stream_.get();
  }

  // Returns a device to host stream. Allocates streams in a round-robin fashion
  // amongst the available streams.
  se::Stream* GetDeviceToHostStream();

  // Returns a device to device stream. Allocates streams in a round-robin
  // fashion amongst the available streams.
  se::Stream* GetDeviceToDeviceStream();

  // Return a stream that should be used to track when an externally-managed
  // buffer is ready. This is intended to support dlpack on GPU. Allocates
  // streams in a round-robin fashion amongst the available streams.
  se::Stream* GetExternalReadyEventStream();

  // Maps a raw platform-specific stream to an se::Stream* owned by this
  // LocalDeviceState. `stream` should have been derived from a se::Stream*
  // returned by GetExternalReadyEventStream.
  // TODO(skyewm): this function could map other raw streams if needed. It's
  // currently only used with external ready event streams.
  StatusOr<se::Stream*> GetStreamFromExternalStream(std::intptr_t stream);

  // Returns a vector of device to device streams.
  std::vector<se::Stream*> GetDeviceToDeviceStreams();

  // Returns a stream from a pool. The stream is guaranteed not to have any
  // currently outstanding work at its tail.
  std::unique_ptr<se::Stream> BorrowStreamFromPool();
  // Returns a stream to the pool. The caller must ensure the stream does not
  // have any outstanding work at its tail.
  void ReturnStreamToPool(std::unique_ptr<se::Stream> stream);

  // Enqueues a copy of `src_buffer` to `dst_buffer` onto `transfer_stream`.
  virtual Status ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                          se::Stream* dst_stream,
                                          se::DeviceMemoryBase src_buffer,
                                          se::DeviceMemoryBase dst_buffer);

  WorkerThread* execute_thread() const { return execute_thread_.get(); }

  // Enqueues a host callback on 'stream'. `stream` may, but need not, wait for
  // `callback` to complete. It is safe to call runtime methods from the
  // callback.
  // This API differs from ThenDoHostCallback in two ways:
  // a) ThenDoHostCallback is often constrained in what it can do, in
  //    particular, on GPU the callback runs on a thread belonging to the GPU
  //    runtime and cannot perform GPU operations itself. On GPU, callbacks
  //    execute in a separate thread.
  // b) ThenDoHostCallback waits for the callback to complete.
  void ThenExecuteCallback(se::Stream* stream, std::function<void()> callback);

  // Helpers for releasing values on a worker thread at the tail of a stream on
  // a worker thread. Copies `object`, and destroys the copy when the tail of
  // the stream is reached. The destruction happens either in the caller's
  // thread or on the worker thread (depending on thread schedules), not a
  // device callback, so it is safe if the destructor frees device resource
  // (e.g., GPU objects).
  template <typename T>
  void ThenRelease(se::Stream* stream, T&& object) {
    ThenExecuteCallback(
        stream, [object = std::forward<T>(object)]() { /* releases object */ });
  }

  Semaphore& compute_semaphore() { return compute_semaphore_; }

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

 private:
  Status SynchronizeAllActivity();

  AllocationModel allocation_model_;

  EventPool event_pool_;

  // Semaphore used to limit how many programs can be enqueued on the compute
  // stream by the host ahead of the device.
  Semaphore compute_semaphore_;

  PjRtLocalDeviceId local_device_id_;
  PjRtLocalHardwareId local_hardware_id_;
  se::StreamExecutor* const executor_;
  LocalClient* const client_;
  std::unique_ptr<se::Stream> compute_stream_;
  std::unique_ptr<se::Stream> host_to_device_stream_;
  std::vector<std::unique_ptr<se::Stream>> device_to_host_streams_;
  std::vector<std::unique_ptr<se::Stream>> device_to_device_streams_;
  std::vector<std::unique_ptr<se::Stream>> external_ready_event_streams_;

  static constexpr int kNumDeviceToHostStreams = 4;
  static constexpr int kNumDeviceToDeviceStreams = 4;
  static constexpr int kNumExternalReadyEventStreams = 4;

  absl::Mutex mu_;
  int next_device_to_host_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_device_to_device_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_external_ready_event_stream_ ABSL_GUARDED_BY(mu_) = 0;
  std::stack<std::unique_ptr<se::Stream>> usage_stream_pool_
      ABSL_GUARDED_BY(mu_);

  std::random_device prng_seed_device_ ABSL_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ ABSL_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ ABSL_GUARDED_BY(mu_);

  // Callback map pairs callback stream with a device stream and is used for
  // running short host-side callbacks after device side events, without
  // preventing the device-side stream from doing useful work.
  absl::Mutex callback_stream_map_mu_;
  std::optional<absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>>
      callback_stream_map_;

  // A worker thread, used for replicated computation launches.
  std::unique_ptr<WorkerThread> execute_thread_;

  // A worker thread, used for callbacks. It is necessary that this be a
  // different thread to the execute thread because we acquire the compute
  // semaphore during calls to Execute but release it from a callback and if
  // they are the same thread we might deadlock.
  std::unique_ptr<WorkerThread> callback_thread_;
};

}  // namespace xla

#endif  // XLA_PJRT_LOCAL_DEVICE_STATE_H_
