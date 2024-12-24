/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// This file extends/implements core stream executor base classes in terms of
// the C API defined in stream_executor.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface and a C
// struct called "SP_Something".
//
// This file also contains stream_executor::Platform registration for pluggable
// device.
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/c_api_macros_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"
#include "tensorflow/core/common_runtime/device/device_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tsl/platform/status.h"

using tensorflow::StatusFromTF_Status;

namespace stream_executor {
using tensorflow::StringPiece;

// TODO(penporn): Remove OwnedTFStatus.
using OwnedTFStatus = tensorflow::TF_StatusPtr;

namespace {
absl::Status ValidateSPPlatform(const SP_Platform& platform) {
  TF_VALIDATE_STRUCT_SIZE(SP_Platform, platform, SP_PLATFORM_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_Platform, platform, name);
  TF_VALIDATE_NOT_NULL(SP_Platform, platform, type);
  TF_RETURN_IF_ERROR(
      tensorflow::device_utils::ValidateDeviceType(platform.name));
  TF_RETURN_IF_ERROR(
      tensorflow::device_utils::ValidateDeviceType(platform.type));
  // `visible_device_count` could be 0 at initialization time.
  return absl::OkStatus();
}

absl::Status ValidateSPPlatformFns(const SP_PlatformFns& platform_fns) {
  TF_VALIDATE_STRUCT_SIZE(SP_PlatformFns, platform_fns,
                          SP_PLATFORM_FNS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_device);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_device);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_stream_executor);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_stream_executor);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_device_fns);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_device_fns);
  return absl::OkStatus();
}

absl::Status ValidateSPAllocatorStats(const SP_AllocatorStats& stats) {
  TF_VALIDATE_STRUCT_SIZE(SP_AllocatorStats, stats,
                          SP_ALLOCATORSTATS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return absl::OkStatus();
}

absl::Status ValidateSPDeviceMemoryBase(const SP_DeviceMemoryBase& mem) {
  TF_VALIDATE_STRUCT_SIZE(SP_DeviceMemoryBase, mem,
                          SP_DEVICE_MEMORY_BASE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return absl::OkStatus();
}

absl::Status ValidateSPDevice(const SP_Device& device) {
  TF_VALIDATE_STRUCT_SIZE(SP_Device, device, SP_DEVICE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return absl::OkStatus();
}

absl::Status ValidateSPDeviceFns(const SP_DeviceFns& device_fns) {
  TF_VALIDATE_STRUCT_SIZE(SP_DeviceFns, device_fns, SP_DEVICE_FNS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return absl::OkStatus();
}

absl::Status ValidateSPStreamExecutor(const SP_StreamExecutor& se,
                                      const SP_Platform& platform) {
  TF_VALIDATE_STRUCT_SIZE(SP_StreamExecutor, se,
                          SP_STREAM_EXECUTOR_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, allocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, deallocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_allocator_stats);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_memory_allocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_memory_deallocate);
  if (platform.supports_unified_memory) {
    TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, unified_memory_allocate);
    TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, unified_memory_deallocate);
  }
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, device_memory_usage);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_stream);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, destroy_stream);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_stream_dependency);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_stream_status);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, destroy_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_event_status);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, record_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, wait_for_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memcpy_dtoh);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memcpy_htod);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, sync_memcpy_dtoh);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, sync_memcpy_htod);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, block_host_for_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, synchronize_all_activity);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_callback);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, mem_zero);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memset);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memset32);
  return absl::OkStatus();
}

absl::Status ValidateSEPlatformRegistrationParams(
    const SE_PlatformRegistrationParams& params) {
  TF_VALIDATE_STRUCT_SIZE(SE_PlatformRegistrationParams, params,
                          SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SE_PlatformRegistrationParams, params, destroy_platform);
  TF_VALIDATE_NOT_NULL(SE_PlatformRegistrationParams, params,
                       destroy_platform_fns);
  return absl::OkStatus();
}
#undef TF_VALIDATE_NOT_NULL

DeviceMemoryBase DeviceMemoryBaseFromC(const SP_DeviceMemoryBase& mem) {
  DeviceMemoryBase base(mem.opaque, mem.size);
  base.SetPayload(mem.payload);
  return base;
}

// Wrapper that allows passing std::function across C API.
struct HostCallbackContext {
  absl::AnyInvocable<absl::Status() &&> callback;
};

// This wrapper allows calling `HostCallbackContext::callback` across C API.
// This function matches `SE_StatusCallbackFn` signature and will be passed as
// `callback_fn` to `host_callback` in `SP_StreamExecutor`.
void HostCallbackTrampoline(void* ctx, TF_Status* status) {
  HostCallbackContext* host_ctx = static_cast<HostCallbackContext*>(ctx);
  absl::Status s = std::move(host_ctx->callback)();
  tsl::Set_TF_Status_from_Status(status, s);
  delete host_ctx;
}

class CStreamExecutor : public StreamExecutorCommon {
 public:
  explicit CStreamExecutor(Platform* se_platform, SP_Device device,
                           SP_DeviceFns* device_fns,
                           SP_StreamExecutor* stream_executor,
                           SP_Platform* platform, SP_PlatformFns* platform_fns,
                           SP_TimerFns* timer_fns, const std::string& name,
                           int visible_device_count)
      : StreamExecutorCommon(se_platform),
        device_(std::move(device)),
        device_fns_(device_fns),
        stream_executor_(stream_executor),
        platform_(platform),
        platform_fns_(platform_fns),
        timer_fns_(timer_fns),
        platform_name_(name),
        visible_device_count_(visible_device_count) {}

  ~CStreamExecutor() override {
    platform_fns_->destroy_device(platform_, &device_);
  }

  absl::Status Init() override { return absl::OkStatus(); }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override {
    SP_DeviceMemoryBase mem = {SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
    stream_executor_->allocate(&device_, size, memory_space, &mem);
    absl::Status status = ValidateSPDeviceMemoryBase(mem);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
    }
    return DeviceMemoryBaseFromC(mem);
  }
  DeviceMemoryBase Allocate(uint64_t size) {
    return Allocate(size, /*memory_space=*/0);
  }

  void Deallocate(DeviceMemoryBase* mem) override {
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(mem);
    stream_executor_->deallocate(&device_, &device_memory_base);
  }

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override {
    auto* buffer = stream_executor_->host_memory_allocate(&device_, size);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(
          absl::StrFormat("Failed to allocate HostMemory of size %d", size));
    }
    return std::make_unique<HostMemoryAllocation>(buffer, size, this);
  }

  void HostMemoryDeallocate(void* mem) override {
    stream_executor_->host_memory_deallocate(&device_, mem);
  }

  void* UnifiedMemoryAllocate(uint64_t size) override {
    CHECK(stream_executor_->unified_memory_allocate);
    return stream_executor_->unified_memory_allocate(&device_, size);
  }

  void UnifiedMemoryDeallocate(void* mem) override {
    CHECK(stream_executor_->unified_memory_deallocate);
    stream_executor_->unified_memory_deallocate(&device_, mem);
  }

  absl::optional<AllocatorStats> GetAllocatorStats() override {
    SP_AllocatorStats c_stats{SP_ALLOCATORSTATS_STRUCT_SIZE};
    TF_Bool has_stats =
        stream_executor_->get_allocator_stats(&device_, &c_stats);
    if (!has_stats) {
      return absl::nullopt;
    }
    absl::Status status = ValidateSPAllocatorStats(c_stats);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
      return absl::nullopt;
    }
    ::stream_executor::AllocatorStats stats;
    stats.num_allocs = c_stats.num_allocs;
    stats.bytes_in_use = c_stats.bytes_in_use;
    stats.peak_bytes_in_use = c_stats.peak_bytes_in_use;
    stats.largest_alloc_size = c_stats.largest_alloc_size;
    if (c_stats.has_bytes_limit) {
      stats.bytes_limit = c_stats.bytes_limit;
    }
    stats.bytes_reserved = c_stats.bytes_reserved;
    stats.peak_bytes_reserved = c_stats.peak_bytes_reserved;
    if (c_stats.has_bytes_reservable_limit) {
      stats.bytes_reservable_limit = c_stats.bytes_reservable_limit;
    }
    stats.largest_free_block_bytes = c_stats.largest_free_block_bytes;
    return stats;
  }
  bool SynchronizeAllActivity() override {
    OwnedTFStatus c_status(TF_NewStatus());
    stream_executor_->synchronize_all_activity(&device_, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override {
    // TODO(annarev): figure out if we should support memzero/memset
    // functionality by allocating on host and then copying to device.
    return tsl::errors::Unimplemented(
        "SynchronousMemZero is not supported by pluggable device.");
  }
  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->sync_memcpy_htod(&device_, &device_memory_base, host_src,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->sync_memcpy_dtoh(&device_, host_dst, &device_memory_base,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  void DeallocateStream(Stream* stream) override {
    static_cast<CStream*>(stream)->Destroy();
  }
  absl::Status BlockHostForEvent(Stream* stream, Event* event) {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Event event_handle = static_cast<CEvent*>(event)->Handle();
    stream_executor_->block_host_for_event(&device_, event_handle,
                                           c_status.get());
    return StatusFromTF_Status(c_status.get());
  }

  absl::Status EnablePeerAccessTo(StreamExecutor* other) override {
    return tsl::errors::Unimplemented(
        "EnablePeerAccessTo is not supported by pluggable device.");
  }
  bool CanEnablePeerAccessTo(StreamExecutor* other) override { return false; }

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override {
    return stream_executor_->device_memory_usage(
        &device_, reinterpret_cast<int64_t*>(free),
        reinterpret_cast<int64_t*>(total));
  }

  // Creates a new DeviceDescription object.
  // Ownership is transferred to the caller.
  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    OwnedTFStatus c_status(TF_NewStatus());

    DeviceDescription desc;
    if (device_.hardware_name != nullptr) {
      desc.set_name(device_.hardware_name);
    }
    if (device_.device_vendor != nullptr) {
      desc.set_device_vendor(device_.device_vendor);
    }
    if (device_.pci_bus_id != nullptr) {
      desc.set_pci_bus_id(device_.pci_bus_id);
    }

    if (device_fns_->get_numa_node != nullptr) {
      int32_t numa_node = device_fns_->get_numa_node(&device_);
      if (numa_node >= 0) {
        desc.set_numa_node(numa_node);
      }
    }

    if (device_fns_->get_memory_bandwidth != nullptr) {
      int64_t memory_bandwidth = device_fns_->get_memory_bandwidth(&device_);
      if (memory_bandwidth >= 0) {
        desc.set_memory_bandwidth(memory_bandwidth);
      }
    }
    // TODO(annarev): Add gflops field in DeviceDescription and set it here.
    // TODO(annarev): Perhaps add `supports_unified_memory` in
    // DeviceDescription.
    return std::make_unique<DeviceDescription>(std::move(desc));
  }

  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override {
    auto c_event = std::make_unique<CEvent>(&device_, stream_executor_);
    TF_RETURN_IF_ERROR(c_event->Create());
    return std::move(c_event);
  }

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override {
    auto stream = std::make_unique<CStream>(&device_, stream_executor_, this);
    TF_RETURN_IF_ERROR(stream->Create());
    return std::move(stream);
  }

 private:
  SP_Device device_;
  SP_DeviceFns* device_fns_;
  SP_StreamExecutor* stream_executor_;
  SP_Platform* platform_;
  SP_PlatformFns* platform_fns_;
  SP_TimerFns* timer_fns_;
  std::string platform_name_;
  int visible_device_count_;
};
}  // namespace

CPlatform::CPlatform(SP_Platform platform,
                     void (*destroy_platform)(SP_Platform*),
                     SP_PlatformFns platform_fns,
                     void (*destroy_platform_fns)(SP_PlatformFns*),
                     SP_DeviceFns device_fns, SP_StreamExecutor stream_executor,
                     SP_TimerFns timer_fns)
    : platform_(std::move(platform)),
      destroy_platform_(destroy_platform),
      platform_fns_(std::move(platform_fns)),
      destroy_platform_fns_(destroy_platform_fns),
      device_fns_(std::move(device_fns)),
      stream_executor_(std::move(stream_executor)),
      timer_fns_(std::move(timer_fns)),
      name_(platform.name) {}

CPlatform::~CPlatform() {
  platform_fns_.destroy_device_fns(&platform_, &device_fns_);
  platform_fns_.destroy_stream_executor(&platform_, &stream_executor_);
  platform_fns_.destroy_timer_fns(&platform_, &timer_fns_);
  destroy_platform_(&platform_);
  destroy_platform_fns_(&platform_fns_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CPlatform::DescriptionForDevice(int ordinal) const {
  // TODO(annarev): see if we can get StreamExecutor instance
  // and call GetDeviceDescription. executor_cache_.Get would need
  // to be made const for it to work.
  DeviceDescription desc;
  desc.set_name(name_);
  return std::make_unique<DeviceDescription>(std::move(desc));
}
absl::StatusOr<StreamExecutor*> CPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}
absl::StatusOr<StreamExecutor*> CPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}
absl::StatusOr<std::unique_ptr<StreamExecutor>> CPlatform::GetUncachedExecutor(
    int ordinal) {
  // Fill device creation params
  SE_CreateDeviceParams device_params{SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE};
  SP_Device device{SP_DEVICE_STRUCT_SIZE};
  device_params.device = &device;
  device_params.ext = nullptr;
  device_params.ordinal = ordinal;
  OwnedTFStatus c_status(TF_NewStatus());

  // Create Device
  platform_fns_.create_device(&platform_, &device_params, c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDevice(device));

  // Get Device Count
  int visible_device_count = 0;
  platform_fns_.get_device_count(&platform_, &visible_device_count,
                                 c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));

  return std::make_unique<CStreamExecutor>(
      this, std::move(device), &device_fns_, &stream_executor_, &platform_,
      &platform_fns_, &timer_fns_, name_, visible_device_count);
}

absl::Status InitStreamExecutorPlugin(void* dso_handle,
                                      std::string* device_type,
                                      std::string* platform_name) {
  tensorflow::Env* env = tensorflow::Env::Default();

  // Step 1: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol));

  // Step 2: Call `TF_InitPlugin`
  auto init_fn = reinterpret_cast<SEInitPluginFn>(dso_symbol);
  return InitStreamExecutorPlugin(init_fn, device_type, platform_name);
}

absl::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn,
                                      std::string* device_type,
                                      std::string* platform_name) {
  SE_PlatformRegistrationParams params{
      SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE};
  SP_Platform platform{SP_PLATFORM_STRUCT_SIZE};
  SP_PlatformFns platform_fns{SP_PLATFORM_FNS_STRUCT_SIZE};
  params.major_version = SE_MAJOR;
  params.minor_version = SE_MINOR;
  params.patch_version = SE_PATCH;
  params.platform = &platform;
  params.platform_fns = &platform_fns;

  OwnedTFStatus c_status(TF_NewStatus());
  init_fn(&params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSEPlatformRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateSPPlatform(platform));
  TF_RETURN_IF_ERROR(ValidateSPPlatformFns(platform_fns));

  // Fill SP_DeviceFns creation params
  SE_CreateDeviceFnsParams device_fns_params{
      SE_CREATE_DEVICE_FNS_PARAMS_STRUCT_SIZE};
  SP_DeviceFns device_fns{SP_DEVICE_FNS_STRUCT_SIZE};
  device_fns_params.device_fns = &device_fns;

  // Create StreamExecutor
  platform_fns.create_device_fns(&platform, &device_fns_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDeviceFns(device_fns));

  // Fill stream executor creation params
  SE_CreateStreamExecutorParams se_params{
      SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE};
  SP_StreamExecutor se{SP_STREAMEXECUTOR_STRUCT_SIZE};
  se_params.stream_executor = &se;

  // Create StreamExecutor
  platform_fns.create_stream_executor(&platform, &se_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPStreamExecutor(se, platform));

  SP_TimerFns timer_fns{SP_TIMER_FNS_STRUCT_SIZE};
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));

  // Register new platform
  *device_type = std::string(platform.type);
  *platform_name = std::string(platform.name);
  std::unique_ptr<stream_executor::CPlatform> cplatform(
      new stream_executor::CPlatform(
          std::move(platform), params.destroy_platform, std::move(platform_fns),
          params.destroy_platform_fns, std::move(device_fns), std::move(se),
          std::move(timer_fns)));
  TF_CHECK_OK(
      stream_executor::PlatformManager::RegisterPlatform(std::move(cplatform)));
  // TODO(annarev): Return `use_bfc_allocator` value in some way so that it is
  // available in `PluggableDeviceProcessState` once the latter is checked in.
  return absl::OkStatus();
}
}  // namespace stream_executor
