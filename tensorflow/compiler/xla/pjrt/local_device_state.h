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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_

#include <memory>
#include <random>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/pjrt/event_pool.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/stream_executor.h"

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

  // If asynchronous is false, the host will synchronize to the device after
  // each execution or transfer. This is intended for debugging only.
  LocalDeviceState(se::StreamExecutor* executor, LocalClient* client,
                   AllocationModel allocation_model,
                   int max_inflight_computations, bool allow_event_reuse,
                   bool use_callback_stream);
  virtual ~LocalDeviceState();

  se::StreamExecutor* executor() const { return executor_; }
  // StreamExecutor (local) device ordinal.
  int device_ordinal() const { return executor_->device_ordinal(); }

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

  se::StreamExecutor* const executor_;
  LocalClient* const client_;
  std::unique_ptr<se::Stream> compute_stream_;
  std::unique_ptr<se::Stream> host_to_device_stream_;
  std::vector<std::unique_ptr<se::Stream>> device_to_host_streams_;
  std::vector<std::unique_ptr<se::Stream>> device_to_device_streams_;

  // Number of device-to-host and device-to-device streams.
  static constexpr int kNumDeviceToHostStreams = 4;
  static constexpr int kNumDeviceToDeviceStreams = 4;

  absl::Mutex mu_;
  int next_device_to_host_stream_ TF_GUARDED_BY(mu_) = 0;
  int next_device_to_device_stream_ TF_GUARDED_BY(mu_) = 0;
  std::stack<std::unique_ptr<se::Stream>> usage_stream_pool_ TF_GUARDED_BY(mu_);

  std::random_device prng_seed_device_ TF_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ TF_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ TF_GUARDED_BY(mu_);

  // Callback map pairs callback stream with a device stream and is used for
  // running short host-side callbacks after device side events, without
  // preventing the device-side stream from doing useful work.
  absl::optional<absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>>
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

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_
