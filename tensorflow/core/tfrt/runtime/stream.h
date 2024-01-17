/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and limitations
under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_STREAM_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_STREAM_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tfrt/runtime/step_id.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool_interface.h"

namespace tensorflow {
namespace tfrt_stub {

struct StreamedResult {
  absl::flat_hash_map<std::string, tensorflow::Tensor> tensors;
  absl::Time enqueued_time;
};

struct StreamCallbackId : SafeId<StreamCallbackId> {
  using Base::Base;
};

// An interface that abstracts communication between the
// `StreamCallbackRegistry` and the stream controller backend.
class StreamControllerInterface {
 public:
  explicit StreamControllerInterface(std::string controller_address)
      : controller_address_(std::move(controller_address)) {}
  virtual ~StreamControllerInterface() = default;

  absl::string_view controller_address() const { return controller_address_; }

  virtual void RecordDequeueLatency(absl::string_view model_name,
                                    absl::Duration latency) {}

  virtual void RecordCallbackLatency(absl::string_view model_name,
                                     absl::Duration latency) {}

 private:
  std::string controller_address_;
};

// An interface that abstracts the communication from the `PwStreamResultsOp`
// worker to the controller.
class StreamWorkerInterface {
 public:
  explicit StreamWorkerInterface(std::string controller_address)
      : controller_address_(std::move(controller_address)) {}
  virtual ~StreamWorkerInterface() = default;

  absl::string_view controller_address() const { return controller_address_; }

  virtual void RecordSendLatency(absl::string_view model_name,
                                 absl::Duration latency) {}
  virtual absl::Status InvokeStreamCallback(
      const StreamCallbackId& callback_id,
      const std::vector<std::string>& names,
      const std::vector<std::pair<int64_t, std::vector<tensorflow::Tensor>>>&
          responses) = 0;

 private:
  std::string controller_address_;
};

class ScopedStreamCallback;

class StreamInterfaceFactory {
 public:
  using CreateWorkerStreamInterfaceFn =
      std::function<absl::StatusOr<std::unique_ptr<StreamWorkerInterface>>(
          absl::string_view)>;

  void RegisterController(
      absl::AnyInvocable<
          absl::StatusOr<std::unique_ptr<StreamControllerInterface>>() const>
          interface_factory) {
    absl::MutexLock lock(&mu_);
    controller_interface_factory_ = std::move(interface_factory);
  }

  absl::StatusOr<std::unique_ptr<StreamControllerInterface>>
  CreateControllerStreamInterface() const {
    absl::MutexLock lock(&mu_);
    return controller_interface_factory_();
  }

  void RegisterWorker(CreateWorkerStreamInterfaceFn interface_factory) {
    absl::MutexLock lock(&mu_);
    worker_interface_factory_ = std::move(interface_factory);
  }

  CreateWorkerStreamInterfaceFn CreateWorkerStreamInterface() const {
    absl::MutexLock lock(&mu_);
    return worker_interface_factory_;
  }

 private:
  mutable absl::Mutex mu_;
  absl::AnyInvocable<
      absl::StatusOr<std::unique_ptr<StreamControllerInterface>>() const>
      controller_interface_factory_ ABSL_GUARDED_BY(mu_) = []() {
        return absl::InternalError(
            "The factory for StreamControllerInterface is not registered.");
      };

  CreateWorkerStreamInterfaceFn worker_interface_factory_ ABSL_GUARDED_BY(mu_) =
      [](absl::string_view) {
        return absl::InternalError(
            "The factory for StreamWorkerInterface is not registered.");
      };
};

// Returns the global factory for the stream interface. The factory for the
// stream interface must be registered first before calling
// GetGlobalStreamCallbackRegistry().
StreamInterfaceFactory& GetGlobalStreamInterfaceFactory();

// Mapping from tuples of (callback_id, step_id) to callback states. The mapping
// is stored in a global variable so that it can be shared between
// `ScopedStreamCallback` and `InvokeStreamCallbackOp`.
//
// This class is thread-safe.
class StreamCallbackRegistry {
 public:
  explicit StreamCallbackRegistry(
      std::unique_ptr<StreamControllerInterface> interface)
      : interface_(std::move(interface)) {
    DCHECK(interface_);
  }

  // Registers a callback under the given id. A stream callback is uniquely
  // identified by a tuple of a callback id (unique to each executable) and a
  // step id (unique to each invocation of a given executable). Returns an RAII
  // object that removes the callback from the registry on its deallocation, or
  // an error if the id already exists in the registry.
  //
  // If a program runs `tf.PwStreamResults` with a matching callback/step id,
  // `callback` will be called with the arguments of `tf.PwStreamResults`.
  //
  // All invocations to `callback` are handled serially by a single thread, so
  // `callback` doesn't need to be thread-safe even if multiple
  // `tf.PwStreamResults` ops may run concurrently.
  absl::StatusOr<ScopedStreamCallback> Register(
      absl::string_view model_name, StreamCallbackId callback_id,
      StepId step_id,
      absl::AnyInvocable<
          void(absl::flat_hash_map<std::string, tensorflow::Tensor>)>
          callback);

  absl::Status Invoke(tsl::thread::ThreadPoolInterface* thread_pool,
                      StreamCallbackId callback_id, StepId step_id,
                      StreamedResult result);

  StreamControllerInterface& stream_interface() const { return *interface_; }

 private:
  friend class ScopedStreamCallback;

  class CallbackState {
   public:
    CallbackState(StreamCallbackRegistry* registry,
                  absl::string_view model_name, StreamCallbackId callback_id,
                  StepId step_id,
                  absl::AnyInvocable<void(
                      absl::flat_hash_map<std::string, tensorflow::Tensor>)>
                      callback)
        : registry_(registry),
          model_name_(model_name),
          callback_id_(callback_id),
          step_id_(step_id),
          callback_(std::move(callback)) {
      DCHECK(registry_);
    }

    // Invokes the callback in `thread_pool` with `result`.
    absl::Status Invoke(tsl::thread::ThreadPoolInterface* thread_pool,
                        StreamedResult result);

    // Closes the callback so that it can no longer be invoked. This method also
    // waits for outstanding results to finish.
    void Close();

   private:
    StreamControllerInterface& interface() {
      return registry_->stream_interface();
    }
    void InvokeCallback(StreamedResult result);

    StreamCallbackRegistry* registry_ = nullptr;
    std::string model_name_;
    StreamCallbackId callback_id_;
    StepId step_id_;
    absl::AnyInvocable<void(
        absl::flat_hash_map<std::string, tensorflow::Tensor>)>
        callback_;

    absl::Mutex mu_;
    bool closed_ ABSL_GUARDED_BY(mu_) = false;
    int num_outstanding_ ABSL_GUARDED_BY(mu_) = 0;
  };

  std::unique_ptr<CallbackState> Unregister(StreamCallbackId callback_id,
                                            StepId step_id);

  std::unique_ptr<StreamControllerInterface> interface_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::pair<StreamCallbackId, StepId>,
                      std::unique_ptr<CallbackState>>
      stream_callbacks_ ABSL_GUARDED_BY(mu_);
};

// Returns the global registry for the stream callbacks. The stream interface
// must have been registered through GetGlobalStreamInterfaceFactory() before
// calling this function.
StreamCallbackRegistry& GetGlobalStreamCallbackRegistry();

// Creates a new stream callback id and rewrites the given module with
// information required to trigger this callback remotely. Returns the callback
// id, or `std::nullopt` if the module has no stream outputs.
absl::StatusOr<std::optional<StreamCallbackId>> CreateStreamCallbackId(
    absl::string_view model_name, mlir::ModuleOp module);

// Implements an RAII object that registers a callback to be called on receiving
// streamed tensors.
class ScopedStreamCallback {
 public:
  ScopedStreamCallback() = default;

  // Moveable but not copyable.
  ScopedStreamCallback(ScopedStreamCallback&& other);
  ScopedStreamCallback& operator=(ScopedStreamCallback&& other);

  ~ScopedStreamCallback() { Unregister(); }

 private:
  friend class StreamCallbackRegistry;

  explicit ScopedStreamCallback(StreamCallbackRegistry* registry,
                                StreamCallbackId callback_id, StepId step_id)
      : registry_(registry), callback_id_(callback_id), step_id_(step_id) {}

  void Unregister();

  StreamCallbackRegistry* registry_ = nullptr;
  std::optional<StreamCallbackId> callback_id_;
  StepId step_id_ = StepId::GetInvalidStepId();
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_STREAM_H_
