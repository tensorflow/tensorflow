/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_context.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/context_map.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

namespace {

// Returns the current context or dies if it fails.
hipCtx_t CurrentContextOrDie() {
  hipCtx_t current = nullptr;
  TF_CHECK_OK(
      ToStatus(hipCtxGetCurrent(&current), "Failed to query current context"));
  return current;
}

// Returns the current context and checks that it is in the set of HIP contexts
// created by StreamExecutor (to ensure that the HIP runtime didn't create a
// context behind our backs).
hipCtx_t CurrentContext() {
  hipCtx_t current = CurrentContextOrDie();
  if (current != nullptr && !RocmContext::GetContextMap()->Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "rocm_driver API: "
               << current
               << "; a HIP runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// Returns the amount of memory reserved by ROCm libraries.
bool GetReservedMemory(uint64_t* reserve) {
  hipDeviceProp_t props;
  hipDevice_t dev;
  hipError_t res = wrap::hipGetDevice(&dev);

  if (res != hipSuccess) {
    LOG(FATAL) << "failed to query current device: " << ToString(res);
    return false;
  }
  res = wrap::hipGetDeviceProperties(&props, dev);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  std::string gcnArchName = props.gcnArchName;
  auto compute_capability = RocmComputeCapability(gcnArchName);
  // On gfx90a, we hide 1 GB of GPU memory (512MB for gfx908) from TF,
  // to allow for late allocations by internal ROCm libraries
  // (e.g. rocBLAS alone needs~200 MB to put its kernels as of ROCm 4.1)
  const uint64_t RESERVED_GFX908 = 1048576 * 512;
  const uint64_t RESERVED_GFX9_X = 1048576 * 1024;
  const uint64_t RESERVED_GFX10_X = 1048576 * 512;
  const uint64_t RESERVED_GFX11_X = 1048576 * 512;
  if (compute_capability.gfx9_mi100()) {
    *reserve = RESERVED_GFX908;
  } else if (compute_capability.gfx9_mi200_or_later()) {
    *reserve = RESERVED_GFX9_X;
  } else if (compute_capability.gfx10_rx68xx() ||
             compute_capability.gfx10_rx69xx()) {
    *reserve = RESERVED_GFX10_X;
  } else if (compute_capability.gfx11_rx7900()) {
    *reserve = RESERVED_GFX11_X;
  }

  return true;
}

// Returns the total amount of memory available for allocation by the ROCM
// context, in bytes, via hipDeviceTotalMem.
bool GetDeviceTotalMemory(hipDevice_t device, uint64_t* result) {
  size_t value = -1;
  hipError_t res = wrap::hipDeviceTotalMem(&value, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }
  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for ROCm libraries";
    return false;
  }
  *result = value - reserve;
  return true;
}

}  // namespace

// Returns the singleton ContextMap.
ContextMap<hipCtx_t, RocmContext>* RocmContext::GetContextMap() {
  static ContextMap<hipCtx_t, RocmContext>* context_map =
      new ContextMap<hipCtx_t, RocmContext>([](void* ptr) {
        int device_ordinal;
        hipError_t result =
            hipPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                   HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                   reinterpret_cast<hipDeviceptr_t>(ptr));
        if (result != hipSuccess) {
          LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                     << ". Error: " << ToString(result);
        }
        return device_ordinal;
      });
  return context_map;
}

bool RocmContext::GetDeviceMemoryUsage(int64_t* free_out, int64_t* total_out) {
  ScopedActivateContext activation(this);
  size_t free = 0;
  size_t total = 0;
  hipError_t res = wrap::hipMemGetInfo(&free, &total);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for ROCm libraries";
    return false;
  }

  VLOG(1) << "Device memory: " << total / 1048576 << " MB total, "
          << free / 1048576 << " MB free, reserving " << reserve / 1048576
          << " MB";

  // overflow check
  if (free > std::numeric_limits<int64_t>::max()) {
    LOG(ERROR) << "free memory (" << free << ") is overflow int64_t";
    return false;
  }

  *free_out = free >= reserve ? free - reserve : 0;
  *total_out = total - reserve;
  return true;
}

RocmContext::~RocmContext() {
  hipCtx_t former_context = CurrentContext();
  // Explicitly call RocmContext::SetActive() to silence clang-tidy warnings
  // about calling a virtual method in the destructor.
  RocmContext::SetActive();
  hipDevice_t device;
  CHECK_EQ(hipSuccess, wrap::hipCtxGetDevice(&device));
  CHECK_EQ(hipSuccess, wrap::hipCtxSetCurrent(former_context));

  auto res = wrap::hipDevicePrimaryCtxRelease(device);

  if (res != hipSuccess) {
    LOG(ERROR) << "failed to release HIP context; leaking: " << ToString(res);
  }

  GetContextMap()->Remove(context());
}

void RocmContext::SetActive() {
  TF_CHECK_OK(
      ToStatus(wrap::hipCtxSetCurrent(context_), "Failed setting context"));
}

bool RocmContext::IsActive() const { return CurrentContext() == context_; }

absl::Status RocmContext::Synchronize() {
  ScopedActivateContext activation(this);
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipDeviceSynchronize(),
                              "could not synchronize on ROCM device"));
  return absl::OkStatus();
}

absl::StatusOr<RocmContext*> RocmContext::Create(int device_ordinal,
                                                 hipDevice_t device) {
  RocmContext* context = nullptr;

  int flags = 0;

  hipError_t res;
  hipCtx_t former_context;
  hipCtx_t new_context;

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  CHECK_EQ(hipSuccess, wrap::hipDevicePrimaryCtxGetState(
                           device, &former_primary_context_flags,
                           &former_primary_context_is_active));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      CHECK_EQ(hipSuccess, wrap::hipDevicePrimaryCtxSetFlags(device, flags));
    }
  }

  former_context = CurrentContextOrDie();
  res = wrap::hipDevicePrimaryCtxRetain(&new_context, device);
  if (former_context != nullptr) {
    hipDevice_t former_device;
    if (wrap::hipCtxGetDevice(&former_device) == hipSuccess) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  CHECK_EQ(hipSuccess, wrap::hipCtxSetCurrent(former_context));

  if (res == hipSuccess) {
    context = GetContextMap()->Add(new_context, device_ordinal);
    CHECK(context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created or reused context " << new_context
            << " for this thread";
    return context;
  }

  std::string message =
      "failed call to hipDevicePrimaryCtxRetain: " + ToString(res);
  if (res == hipErrorOutOfMemory) {
    uint64_t total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      absl::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      absl::StrAppend(&message, "; could not query total memory");
    }
  }

  return absl::InternalError(message);
}

}  // namespace stream_executor::gpu
