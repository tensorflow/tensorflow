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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

class GpuStream;

// Intermediate implementation class for StreamExecutors that are used with
// GPUs.
class GpuExecutor : public StreamExecutorCommon {
  // Helper classes to attach a type erased state to the GpuExecutor. Currently,
  // we just need to support some XLA specific state.
  class Object {
    struct Concept {
      virtual ~Concept() {}
    };
    template <typename T>
    struct Model : Concept {
      explicit Model(StreamExecutor* se) : object(se) {}
      T object;
    };

   public:
    template <typename T>
    T* getOrCreate(StreamExecutor* se) {
      absl::MutexLock l(&mu_);
      if (!object_) {
        object_ = std::make_unique<Model<T>>(se);
      }
      return &(dynamic_cast<Model<T>*>(object_.get())->object);
    }

   private:
    absl::Mutex mu_;
    std::unique_ptr<Concept> object_ ABSL_GUARDED_BY(mu_);
  };

 public:
  GpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform),
        context_(nullptr),
        device_ordinal_(device_ordinal) {}

  int device_ordinal() const override { return device_ordinal_; };

  // Releases any state associated with the previously loaded kernel.
  virtual void UnloadKernel(const Kernel* kernel) = 0;
  // Creates an EventBasedTimer for the given stream.
  virtual absl::StatusOr<std::unique_ptr<EventBasedTimer>>
  CreateEventBasedTimer(GpuStream* stream, bool use_delay_kernel) = 0;
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  // Frees unused memory cached on the device for use with graphs back to the
  // OS.
  virtual absl::Status TrimGraphMemory() = 0;

  Context* gpu_context() const { return context_; }

  // Provide a type-erased way of attaching arbitrary XLA specific state to the
  // GpuExecutor. XLA based execution will use this method to attach per-stream
  // executor XLA specific objects (like the Infeed and Outfeed managers) to the
  // stream executor, so that their lifetimes can be tied to the lifetime of the
  // stream executor for which that object is allocated for. This simplifies
  // memory management as compared to having these objects reside on the side
  // and then either leaking or having to implement callbacks that the SE
  // destructors call to deallocate any side state that is associated with that
  // SE object.
  template <typename T>
  T* getOrCreateXLAState(StreamExecutor* se) {
    return xla_state_.getOrCreate<T>(se);
  }

  absl::StatusOr<std::vector<ApiTrace>> ExtractApiTrace() override {
    absl::MutexLock lock(&logger_mu_);
    return std::move(argument_logs_);
  }

  absl::Status RecordApiTrace(ApiTrace call) override {
    absl::MutexLock lock(&logger_mu_);
    if (std::holds_alternative<GemmCallTrace>(call) &&
        (argument_logging_mode_ & kLogGemm)) {
      argument_logs_.push_back(call);
    }
    return absl::OkStatus();
  }

  bool SetArgumentLoggingMode(uint64_t mode) override {
    absl::MutexLock lock(&logger_mu_);
    argument_logging_mode_ = mode;
    return true;
  }

  uint64_t GetArgumentLoggingMode() const { return argument_logging_mode_; }

 protected:
  // Sets the context.
  void set_context(Context* context) { context_ = context; }

 private:
  // Handle for session with the library/driver. Immutable post-initialization.
  Context* context_;

  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  // Type erased XLA specific state attached to GpuExecutor.
  Object xla_state_;

  absl::Mutex logger_mu_;

  mutable std::vector<ApiTrace> argument_logs_ ABSL_GUARDED_BY(logger_mu_);

  uint64_t argument_logging_mode_ = 0;

  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};

inline GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec);
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
