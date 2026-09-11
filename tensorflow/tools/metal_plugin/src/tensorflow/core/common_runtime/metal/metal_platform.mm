/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/metal_platform.h"

#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"
#include "tensorflow/core/common_runtime/metal/metal_stream_executor.h"

namespace tensorflow {
namespace metal {
namespace {

// One usable Metal device, discovered once at startup.
struct DeviceEntry {
  id<MTLDevice> device;  // Retained for the lifetime of the process.
  std::string name;      // Stable storage for SP_Device::hardware_name.
};

// Metal devices this backend is willing to drive.
//
// Only devices reporting hasUnifiedMemory are accepted, and that is a
// correctness gate rather than a preference. The backend hands core the
// `contents` pointer of MTLResourceStorageModeShared buffers and treats host
// and device views of that memory as coherent. On a unified memory device
// (every Apple silicon Mac) they are. On a discrete GPU they are not: shared
// buffers are host-side staging that needs explicit didModifyRange: and
// synchronizeResource: calls, so the zero-copy transfers would silently read
// stale data. Refusing such devices is the honest outcome.
const std::vector<DeviceEntry>& UsableDevices() {
  static const std::vector<DeviceEntry>* devices = [] {
    ScopedAutoreleasePool pool;
    auto* result = new std::vector<DeviceEntry>();
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    for (id<MTLDevice> device in all) {
      const char* name = device.name.UTF8String;
      if (!device.hasUnifiedMemory) {
        LOG(INFO) << "Metal: skipping device '" << (name ? name : "?")
                  << "': the Metal backend requires unified memory.";
        continue;
      }
      result->push_back(
          DeviceEntry{[device retain], name != nullptr ? name : "Metal GPU"});
    }
    [all release];
    return result;
  }();
  return *devices;
}

void Ok(TF_Status* status) { TF_SetStatus(status, TF_OK, ""); }

/*** DEVICE ***/

void GetDeviceCount(const SP_Platform* platform, int* device_count,
                    TF_Status* status) {
  *device_count = static_cast<int>(UsableDevices().size());
  Ok(status);
}

void CreateDevice(const SP_Platform* platform, SE_CreateDeviceParams* params,
                  TF_Status* status) {
  const std::vector<DeviceEntry>& devices = UsableDevices();
  const int ordinal = params->ordinal;
  if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
    TF_SetStatus(status, TF_OUT_OF_RANGE,
                 "Metal: requested device ordinal is out of range.");
    return;
  }
  const DeviceEntry& entry = devices[ordinal];

  params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
  params->device->ordinal = ordinal;
  params->device->device_handle = static_cast<void*>(entry.device);
  // Points into the process-lifetime DeviceEntry, satisfying the C API's
  // requirement that these strings outlive the call.
  params->device->hardware_name = entry.name.c_str();
  params->device->device_vendor = "Apple";
  // Optional, and meaningless for an integrated GPU with no PCI address.
  params->device->pci_bus_id = nullptr;
  params->device->ext = new MetalDeviceState(entry.device);
  Ok(status);
}

void DestroyDevice(const SP_Platform* platform, SP_Device* device) {
  delete StateOf(device);
  device->ext = nullptr;
  device->device_handle = nullptr;
}

int32_t GetNumaNode(const SP_Device* device) {
  // Apple silicon presents a single memory domain, so there is no NUMA node
  // to report and core should treat locality as unset.
  return -1;
}

int64_t GetMemoryBandwidth(const SP_Device* device) {
  // Metal exposes no bandwidth figure, and it differs by an order of magnitude
  // across the M-series range. Reporting a made-up number would skew the cost
  // model, so leave it unset.
  return -1;
}

double GetGflops(const SP_Device* device) { return -1.0; }

void CreateDeviceFns(const SP_Platform* platform,
                     SE_CreateDeviceFnsParams* params, TF_Status* status) {
  params->device_fns->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  params->device_fns->ext = nullptr;
  params->device_fns->get_numa_node = GetNumaNode;
  params->device_fns->get_memory_bandwidth = GetMemoryBandwidth;
  params->device_fns->get_gflops = GetGflops;
  Ok(status);
}

void DestroyDeviceFns(const SP_Platform* platform, SP_DeviceFns* device_fns) {}

/*** STREAM EXECUTOR ***/

void CreateStreamExecutor(const SP_Platform* platform,
                          SE_CreateStreamExecutorParams* params,
                          TF_Status* status) {
  PopulateStreamExecutor(params->stream_executor);
  Ok(status);
}

void DestroyStreamExecutor(const SP_Platform* platform,
                           SP_StreamExecutor* stream_executor) {}

/*** TIMER ***/

void CreateTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                    TF_Status* status) {
  PopulateTimerFns(timer_fns);
  Ok(status);
}

void DestroyTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns) {}

/*** PLATFORM ***/

void DestroyPlatform(SP_Platform* platform) {}

void DestroyPlatformFns(SP_PlatformFns* platform_fns) {}

}  // namespace

void MetalInitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->ext = nullptr;
  // Both are string literals with static storage duration, as the C API
  // requires them to outlive this call.
  params->platform->name = kMetalPlatformName;
  params->platform->type = kMetalDeviceType;
  // True on every device we accept, by construction: UsableDevices() filters
  // out anything without unified memory.
  params->platform->supports_unified_memory = true;
  // Let core put its BFC allocator in front of us. Metal's own allocator is
  // not cheap enough to call per tensor, and BFC also gives the backend the
  // sub-allocation behaviour the buffer registry was designed to support.
  params->platform->use_bfc_allocator = true;
  // On a unified memory system, device memory is the machine's RAM. Reserving
  // a large fraction of it up front, as the CUDA default does, would starve
  // everything else running on the Mac.
  params->platform->force_memory_growth = true;

  params->platform_fns->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
  params->platform_fns->ext = nullptr;
  params->platform_fns->get_device_count = GetDeviceCount;
  params->platform_fns->create_device = CreateDevice;
  params->platform_fns->destroy_device = DestroyDevice;
  params->platform_fns->create_device_fns = CreateDeviceFns;
  params->platform_fns->destroy_device_fns = DestroyDeviceFns;
  params->platform_fns->create_stream_executor = CreateStreamExecutor;
  params->platform_fns->destroy_stream_executor = DestroyStreamExecutor;
  params->platform_fns->create_timer_fns = CreateTimerFns;
  params->platform_fns->destroy_timer_fns = DestroyTimerFns;

  params->destroy_platform = DestroyPlatform;
  params->destroy_platform_fns = DestroyPlatformFns;

  TF_SetStatus(status, TF_OK, "");
}

}  // namespace metal
}  // namespace tensorflow
