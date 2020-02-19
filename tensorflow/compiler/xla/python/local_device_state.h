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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_DEVICE_STATE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_DEVICE_STATE_H_

#include <memory>
#include <random>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/python/event_pool.h"
#include "tensorflow/compiler/xla/python/semaphore.h"
#include "tensorflow/compiler/xla/python/worker_thread.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace xla {

// Class that encapsulates state relating to a device (e.g., a GPU) on which we
// can perform computation and transfers. LocalDeviceState objects only exist
// for devices local to this host.
class LocalDeviceState {
 public:
  // If synchronous_deallocation is true, the host must not free buffers until
  // compute/transfers that use those buffers have completed. For example, this
  // typically is the case for the "platform" where compute/transfers are
  // operations that take place on another thread.
  //
  // If asynchronous is false, the host will synchronize to the device after
  // each execution or transfer. This is intended for debugging only.
  LocalDeviceState(se::StreamExecutor* executor, LocalClient* client,
                   bool synchronous_deallocation, bool asynchronous,
                   bool allow_event_reuse);
  virtual ~LocalDeviceState();

  se::StreamExecutor* executor() const { return executor_; }
  // StreamExecutor (local) device ordinal.
  int device_ordinal() const { return executor_->device_ordinal(); }

  LocalClient* client() const { return client_; }

  bool synchronous_deallocation() const { return synchronous_deallocation_; }

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

  // Enqueues a copy of `src_buffer` to `dst_buffer` onto `transfer_stream`.
  virtual Status ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                          se::Stream* dst_stream,
                                          se::DeviceMemoryBase src_buffer,
                                          se::DeviceMemoryBase dst_buffer);

  WorkerThread* execute_thread() const { return execute_thread_.get(); }

  // Enqueues a host callback on 'stream', to be executed by callback_thread_.
  // ThenDoHostCallback is often constrained in what it can do, in particular,
  // on GPU the callback runs on a thread belonging to the GPU runtime and
  // cannot perform GPU operations itself.
  void ThenExecuteOnCallbackThread(se::Stream* stream,
                                   std::function<void()> callback) const;

  // Helpers for releasing values on a worker thread at the tail of a stream on
  // a worker thread. Copies `object`, and destroys the copy when the tail of
  // the stream is reached. The destruction happens either in the caller's
  // thread or on the worker thread (depending on thread schedules), not a
  // device callback, so it is safe if the destructor frees device resource
  // (e.g., GPU objects).
  // TODO(phawkins): use move-capture when we can use C++14 features.
  template <typename T>
  void ThenRelease(se::Stream* stream, T object) const {
    if (callback_stream_.get() != stream) {
      callback_stream_->ThenWaitFor(stream);
    }
    ThenExecuteOnCallbackThread(callback_stream_.get(),
                                [object]() { /* releases object */ });
  }

  Semaphore& compute_semaphore() { return compute_semaphore_; }

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

 private:
  Status SynchronizeAllActivity();

  bool synchronous_deallocation_;

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
  int next_device_to_host_stream_ GUARDED_BY(mu_) = 0;
  int next_device_to_device_stream_ GUARDED_BY(mu_) = 0;

  std::random_device prng_seed_device_ GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ GUARDED_BY(mu_);

  // Callback stream is used for running short host-side callbacks after device
  // side events, without preventing the device-side stream from doing useful
  // work.
  std::unique_ptr<se::Stream> callback_stream_;

  // A worker thread, used for replicated computation launches.
  std::unique_ptr<WorkerThread> execute_thread_;

  // A worker thread, used for callbacks. It is necessary that this be a
  // different thread to the execute thread because we acquire the compute
  // semaphore during calls to Execute but release it from a callback and if
  // they are the same thread we might deadlock.
  std::unique_ptr<WorkerThread> callback_thread_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_DEVICE_STATE_H_
