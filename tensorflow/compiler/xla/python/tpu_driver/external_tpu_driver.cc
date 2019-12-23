// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <dlfcn.h>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/python/tpu_driver/client/c_api.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tpu_driver {
namespace {

class ExternalTpuDriver;

class ExternalEvent : public Event {
 public:
  explicit ExternalEvent(::TpuDriverFn* driver_fn, ::TpuEvent* event)
      : driver_fn_(driver_fn), event_(event) {}

  ~ExternalEvent() override { driver_fn_->TpuDriver_FreeEvent(event_); }

  xla::Status Await() override {
    auto tpu_status = driver_fn_->TpuDriver_EventAwait(event_, -1);
    auto ret = xla::Status(tensorflow::error::Code(tpu_status->code),
                           absl::StrFormat("%s", tpu_status->msg));
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    return ret;
  }

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    auto tpu_status_or = driver_fn_->TpuDriver_EventAwait(
        event_, absl::ToInt64Microseconds(duration));
    if (tpu_status_or == nullptr) {
      return absl::nullopt;
    } else {
      auto ret = xla::Status(tensorflow::error::Code(tpu_status_or->code),
                             absl::StrFormat("%s", tpu_status_or->msg));
      driver_fn_->TpuDriver_FreeStatus(tpu_status_or);
      return ret;
    }
  }

  void AddCallback(std::function<void(xla::Status)> callback) override {
    // We have to create a new copy of the fn on the heap to make it persist.
    std::function<void(xla::Status)>* callback_addr =
        new std::function<void(xla::Status)>(callback);

    // Using the callback_addr instead of capturing because C++11 lambdas with
    // variable captures cannot be converted to C function pointers.
    driver_fn_->TpuDriver_EventAddCallback(
        event_,
        [](struct TpuStatus* status, void* additional_info) {
          auto callback_addr =
              static_cast<std::function<void(xla::Status)>*>(additional_info);
          auto xla_status = xla::Status(tensorflow::error::Code(status->code),
                                        absl::StrFormat("%s", status->msg));
          (*callback_addr)(xla_status);
          delete callback_addr;
        },
        callback_addr);
  }

 private:
  ::TpuDriverFn* driver_fn_;
  ::TpuEvent* event_;

  friend ExternalTpuDriver;
};

class ExternalBufferHandle : public BufferHandle {
 public:
  explicit ExternalBufferHandle(::TpuDriverFn* driver_fn,
                                ::TpuBufferHandle* handle)
      : handle_(handle), event_(new ExternalEvent(driver_fn, handle->event)) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override { return handle_->size_in_bytes; }

  absl::optional<xla::ShapeProto> shape() override {
    LOG(FATAL) << "Unimplemented.";
    return absl::nullopt;
  }

 private:
  ::TpuBufferHandle* handle_;
  std::shared_ptr<ExternalEvent> event_;

  friend ExternalTpuDriver;
};

class ExternalCompiledProgramHandle : public CompiledProgramHandle {
 public:
  std::shared_ptr<Event> OnReady() override {
    LOG(FATAL) << "Unimplemented";
    return std::shared_ptr<Event>();
  }

  int64_t size_in_bytes() override {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override {
    LOG(FATAL) << "Unimplemented.";
    return xla::Unimplemented("%s", "Unimplemented.");
  }
};

class ExternalLoadedProgramHandle : public LoadedProgramHandle {
 public:
  std::shared_ptr<Event> OnReady() override {
    LOG(FATAL) << "Unimplemented";
    return std::shared_ptr<Event>();
  }

  int64_t size_in_bytes() override {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }
};

class ExternalTpuDriver : public TpuDriver {
 public:
  explicit ExternalTpuDriver(const std::string& so_path) {
    void* handle;
    handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
      LOG(FATAL) << "Unable to load shared library: " << dlerror();
    }

    PrototypeTpuDriver_Initialize* initialize_fn;
    *reinterpret_cast<void**>(&initialize_fn) =
        dlsym(handle, "TpuDriver_Initialize");
    initialize_fn(&driver_fn_);

    driver_ = driver_fn_.TpuDriver_Open("local://");
  }

  ~ExternalTpuDriver() override {}

  void QuerySystemInfo(SystemInfo* system_info) override {
    LOG(FATAL) << "Unimplemented.";
  }

  xla::Status Reset() override { LOG(FATAL) << "Unimplemented."; }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto bh = absl::make_unique<ExternalBufferHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_Allocate(driver_, core_id, region, num_bytes,
                                      wait_for.size(), tpu_events));
    delete tpu_events;
    return bh;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<ExternalEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_Deallocate(
            driver_, static_cast<ExternalBufferHandle*>(handle.get())->handle_,
            wait_for.size(), tpu_events));
    delete tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<ExternalEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferToDevice(
            driver_, src, static_cast<ExternalBufferHandle*>(dst)->handle_,
            wait_for.size(), tpu_events));
    delete tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<ExternalEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDevice(
            driver_, static_cast<const ExternalBufferHandle*>(src)->handle_,
            dst, wait_for.size(), tpu_events));
    delete tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<ExternalEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDeviceToDevice(
            driver_, static_cast<const ExternalBufferHandle*>(src)->handle_,
            static_cast<ExternalBufferHandle*>(dst)->handle_, wait_for.size(),
            tpu_events));
    delete tpu_events;
    return event;
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }
  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }
  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    LOG(FATAL) << "Unimplemented.";
    return nullptr;
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override { return nullptr; }

 private:
  ::TpuDriverFn driver_fn_;
  ::TpuDriver* driver_;

  ::TpuEvent** MakeEventArray(absl::Span<Event* const> wait_for) {
    if (wait_for.empty()) return nullptr;
    ::TpuEvent** ret = new ::TpuEvent*[wait_for.size()];
    for (int i = 0; i < wait_for.size(); i++) {
      ret[i] = static_cast<ExternalEvent* const>(wait_for[i])->event_;
    }
    return ret;
  }
};

xla::StatusOr<std::unique_ptr<TpuDriver>> RegisterExternalTpuDriver(
    const TpuDriverConfig& config) {
  std::string shared_lib = config.worker().substr(strlen("external://"));
  return xla::StatusOr<std::unique_ptr<TpuDriver>>(
      absl::make_unique<ExternalTpuDriver>(shared_lib));
}

REGISTER_TPU_DRIVER("external://", RegisterExternalTpuDriver);

}  // namespace
}  // namespace tpu_driver
