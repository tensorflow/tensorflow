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
#include "tensorflow/compiler/xla/python/tpu_driver/client/libtpu.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tpu_driver {
namespace {

// Enable the macro by default in the Google internal environment where the
// libtpu.so is linked in statically.
#ifdef PLATFORM_GOOGLE
#define TPU_SHARED_LIBRARY_COMPILE_LINK 1
#endif

xla::Status CreateXlaStatus(::TpuStatus* status) {
  if (status->code == tensorflow::error::OK) {
    return xla::Status::OK();
  } else {
    return xla::Status(tensorflow::error::Code(status->code),
                       absl::StrFormat("%s", status->msg));
  }
}

constexpr char kDirectProtocol[] = "direct://";

::TpuAllocationShape GetTpuAllocationShape(const xla::ShapeProto& shape) {
  ::TpuAllocationShape shape_;
  shape_.size = shape.ByteSizeLong();
  shape_.bytes = malloc(shape_.size);
  if (!shape.SerializeToArray(shape_.bytes, shape_.size)) {
    LOG(ERROR) << "Unable to serialize shape to array.";
    free(shape_.bytes);
    shape_.size = 0;
    shape_.bytes = nullptr;
  }
  return shape_;
}

class DirectTpuDriver;

class DirectEvent : public Event {
 public:
  explicit DirectEvent(::TpuDriverFn* driver_fn, ::TpuEvent* event)
      : driver_fn_(driver_fn), event_(event) {}

  ~DirectEvent() override { driver_fn_->TpuDriver_FreeEvent(event_); }

  xla::Status Await() override {
    auto tpu_status = driver_fn_->TpuDriver_EventAwait(event_, -1);
    auto ret = CreateXlaStatus(tpu_status);
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
      auto ret = CreateXlaStatus(tpu_status_or);
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
          auto xla_status = CreateXlaStatus(status);
          (*callback_addr)(xla_status);
          delete callback_addr;
        },
        callback_addr);
  }

 private:
  ::TpuDriverFn* driver_fn_;
  ::TpuEvent* event_;

  friend DirectTpuDriver;
};

class DirectBufferHandle : public BufferHandle {
 public:
  explicit DirectBufferHandle(::TpuDriverFn* driver_fn,
                              ::TpuBufferHandle* handle)
      : handle_(handle), event_(new DirectEvent(driver_fn, handle->event)) {}

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override { return handle_->size_in_bytes; }

  absl::optional<xla::ShapeProto> shape() override {
    LOG(FATAL) << "Unimplemented.";
    return absl::nullopt;
  }

 private:
  ::TpuBufferHandle* handle_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit DirectCompiledProgramHandle(::TpuDriverFn* driver_fn,
                                       ::TpuCompiledProgramHandle* handle)
      : handle_(handle),
        driver_fn_(driver_fn),
        event_(new DirectEvent(driver_fn, handle->event)) {}

  ~DirectCompiledProgramHandle() override {
    driver_fn_->TpuDriver_FreeCompiledProgramHandle(handle_);
  }

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override {
    struct CompiledProgramShape* shape =
        driver_fn_->TpuDriver_GetCompiledProgramShape(handle_);
    program_shape->ParseFromArray(shape->bytes, shape->size);

    auto status = CreateXlaStatus(shape->status);
    driver_fn_->TpuDriver_FreeCompiledProgramShape(shape);
    return status;
  }

 private:
  ::TpuCompiledProgramHandle* handle_;
  ::TpuDriverFn* driver_fn_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit DirectLoadedProgramHandle(::TpuDriverFn* driver_fn,
                                     ::TpuLoadedProgramHandle* handle)
      : handle_(handle), event_(new DirectEvent(driver_fn, handle->event)) {}
  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

 private:
  ::TpuLoadedProgramHandle* handle_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectTpuLinearizer : public TpuLinearizer {
 public:
  explicit DirectTpuLinearizer(::TpuDriver* driver, ::TpuDriverFn* driver_fn)
      : driver_(driver), driver_fn_(driver_fn) {}

  int64_t ComputeLinearizedBytesFromShape(
      const xla::ShapeProto& shape) override {
    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);
    uint64_t size =
        driver_fn_->TpuDriver_ComputeLinearizedBytesFromShape(driver_, shape_);
    free(shape_.bytes);
    return size;
  }

  xla::Status LinearizeShape(void* dst, const void* src,
                             const xla::ShapeProto& shape) override {
    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);

    auto tpu_status =
        driver_fn_->TpuDriver_LinearizeShape(driver_, dst, src, shape_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    free(shape_.bytes);
    return status;
  }

  xla::Status DelinearizeShape(void* dst, const void* src,
                               const xla::ShapeProto& shape) override {
    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);

    auto tpu_status =
        driver_fn_->TpuDriver_DelinearizeShape(driver_, dst, src, shape_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    free(shape_.bytes);
    return status;
  }

 private:
  ::TpuDriver* driver_;
  ::TpuDriverFn* driver_fn_;
};

class DirectTpuDriver : public TpuDriver {
 public:
  explicit DirectTpuDriver(const std::string& so_path) {
    void* handle;
    handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
      LOG(FATAL) << "Unable to load shared library: " << dlerror();
    }

    PrototypeTpuDriver_Initialize* initialize_fn;
    *reinterpret_cast<void**>(&initialize_fn) =
        dlsym(handle, "TpuDriver_Initialize");
    initialize_fn(&driver_fn_, /*initialize=*/true);

    driver_ = driver_fn_.TpuDriver_Open("local://");
  }

#ifdef TPU_SHARED_LIBRARY_COMPILE_LINK
  DirectTpuDriver() {
    TpuDriver_Initialize(&driver_fn_, /*initialize=*/false);
    driver_ = driver_fn_.TpuDriver_Open("local://");
  }
#endif

  ~DirectTpuDriver() override { driver_fn_.TpuDriver_Close(driver_); }

  void QuerySystemInfo(SystemInfo* system_info) override {
    ::TpuSystemInfo* info = driver_fn_.TpuDriver_QuerySystemInfo(driver_);
    system_info->ParseFromArray(info->bytes, info->size);
    driver_fn_.TpuDriver_FreeSystemInfo(info);
  }

  xla::Status Reset() override {
    auto tpu_status = driver_fn_.TpuDriver_Reset(driver_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_.TpuDriver_FreeStatus(tpu_status);
    return status;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_Allocate(driver_, core_id, region, num_bytes,
                                      wait_for.size(), tpu_events));
    delete[] tpu_events;
    return bh;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);
    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_AllocateShape(driver_, core_id, region, shape_,
                                           wait_for.size(), tpu_events));

    free(shape_.bytes);
    delete[] tpu_events;
    return bh;
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    ::TpuBufferHandle** childbuf = new ::TpuBufferHandle*[children.size()];
    for (int i = 0; i < children.size(); i++) {
      childbuf[i] =
          static_cast<DirectBufferHandle* const>(children[i])->handle_;
    }

    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_, driver_fn_.TpuDriver_AllocateTuple(
                         driver_, core_id, region, children.size(), childbuf,
                         wait_for.size(), tpu_events));
    delete[] tpu_events;
    delete[] childbuf;

    return bh;
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto* direct_bh = static_cast<DirectBufferHandle*>(handle.get());
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_Deallocate(driver_, direct_bh->handle_,
                                        wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferToDevice(
            driver_, src, static_cast<DirectBufferHandle*>(dst)->handle_,
            wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDevice(
            driver_, static_cast<const DirectBufferHandle*>(src)->handle_, dst,
            wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDeviceToDevice(
            driver_, static_cast<const DirectBufferHandle*>(src)->handle_,
            static_cast<DirectBufferHandle*>(dst)->handle_, wait_for.size(),
            tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    struct HloProto hlo;
    hlo.size = source.ByteSizeLong();
    hlo.buffer = malloc(hlo.size);
    if (!source.SerializeToArray(hlo.buffer, hlo.size)) {
      LOG(ERROR) << "Unable to serialize HLO to array.";
      return nullptr;
    }

    auto handle = absl::make_unique<DirectCompiledProgramHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_CompileProgram(driver_, hlo, num_replicas,
                                            wait_for.size(), tpu_events));

    free(hlo.buffer);
    delete[] tpu_events;
    return handle;
  }
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    auto loaded_handle = absl::make_unique<DirectLoadedProgramHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_LoadProgram(
            driver_, core_id,
            static_cast<const DirectCompiledProgramHandle*>(handle)->handle_,
            wait_for.size(), tpu_events));

    delete[] tpu_events;
    return loaded_handle;
  }

  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto* direct_lph = static_cast<DirectLoadedProgramHandle*>(handle.get());
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_UnloadProgram(driver_, direct_lph->handle_,
                                           wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    std::vector<::TpuBufferHandle*> inputv;
    inputv.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      inputv.push_back(
          static_cast<DirectBufferHandle* const>(inputs[i])->handle_);
    }
    std::vector<::TpuBufferHandle*> outputv;
    outputv.reserve(outputs.size());
    for (int i = 0; i < outputs.size(); i++) {
      outputv.push_back(
          static_cast<DirectBufferHandle* const>(outputs[i])->handle_);
    }

    struct DeviceAssignment da;
    da.size = device_assignment.ByteSizeLong();
    da.bytes = malloc(da.size);
    device_assignment.SerializeToArray(da.bytes, da.size);

    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_ExecuteProgram(
            driver_, static_cast<DirectLoadedProgramHandle*>(program)->handle_,
            inputs.size(), inputv.data(), outputs.size(), outputv.data(), da,
            wait_for.size(), tpu_events));

    free(da.bytes);
    delete[] tpu_events;
    return event;
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override {
    return std::make_unique<DirectTpuLinearizer>(driver_, &driver_fn_);
  }

 private:
  ::TpuDriverFn driver_fn_;
  ::TpuDriver* driver_;

  ::TpuEvent** MakeEventArray(absl::Span<Event* const> wait_for) {
    if (wait_for.empty()) return nullptr;
    ::TpuEvent** ret = new ::TpuEvent*[wait_for.size()];
    for (int i = 0; i < wait_for.size(); i++) {
      ret[i] = static_cast<DirectEvent* const>(wait_for[i])->event_;
    }
    return ret;
  }
};

xla::StatusOr<std::unique_ptr<TpuDriver>> RegisterDirectTpuDriver(
    const TpuDriverConfig& config) {
  std::string shared_lib = config.worker().substr(strlen(kDirectProtocol));
  if (shared_lib == "internal") {
#ifdef TPU_SHARED_LIBRARY_COMPILE_LINK
    return xla::StatusOr<std::unique_ptr<TpuDriver>>(
        absl::make_unique<DirectTpuDriver>());
#else
    LOG(FATAL) << "Request to use compile-time linked TPU library, but did not "
               << "link in appropriate library at compile time.";
#endif
  }
  return xla::StatusOr<std::unique_ptr<TpuDriver>>(
      absl::make_unique<DirectTpuDriver>(shared_lib));
}

REGISTER_TPU_DRIVER(kDirectProtocol, RegisterDirectTpuDriver);

}  // namespace
}  // namespace tpu_driver
