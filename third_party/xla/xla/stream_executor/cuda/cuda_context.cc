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

#include "xla/stream_executor/cuda/cuda_context.h"

#include <cstdlib>
#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/context_map.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

namespace {

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

int GetFlagsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return 0;
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_SPIN;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_YIELD;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_BLOCKING_SYNC;
  } else {
    LOG(QFATAL) << "Unknown option for environment variable "
                   "TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE "
                << gpu_schedule_string << " should be one of {"
                << kScheduleBlockingSyncString << ", " << kScheduleSpinString
                << ", " << kScheduleYieldString << "}";
  }

  return device_flags;
}

// Returns the current context or dies if it fails.
CUcontext CurrentContextOrDie() {
  CUcontext current = nullptr;
  TF_CHECK_OK(cuda::ToStatus(cuCtxGetCurrent(&current),
                             "Failed to query current context"));
  return current;
}

// Returns the current context and checks that it is in the set of CUDA contexts
// created by StreamExecutor (to ensure that the CUDA runtime didn't create a
// context behind our backs).
CUcontext CurrentContext() {
  CUcontext current = CurrentContextOrDie();
  if (current != nullptr && !CudaContext::GetContextMap()->Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "cuda_driver API: "
               << current
               << "; a CUDA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

}  // namespace

// Returns the singleton ContextMap.
ContextMap<CUcontext, CudaContext>* CudaContext::GetContextMap() {
  static ContextMap<CUcontext, CudaContext>* context_map =
      new ContextMap<CUcontext, CudaContext>([](void* ptr) {
        int device_ordinal;
        absl::Status status = cuda::ToStatus(
            cuPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  reinterpret_cast<CUdeviceptr>(ptr)));
        if (!status.ok()) {
          LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                     << ". Error: " << status;
        }
        return device_ordinal;
      });
  return context_map;
}

CudaContext::~CudaContext() {
  auto status = cuda::ToStatus(cuCtxPushCurrent(context()));
  if (!status.ok()) {
    LOG(ERROR) << "failed to Push CUDA context; leaking: " << status;
  }
  CUdevice device;
  cuCtxGetDevice(&device);
  cuCtxPopCurrent(nullptr);

  status = cuda::ToStatus(cuDevicePrimaryCtxRelease(device));

  if (!status.ok()) {
    LOG(ERROR) << "failed to release CUDA context; leaking: " << status;
  }

  GetContextMap()->Remove(context());
}

absl::StatusOr<CudaContext*> CudaContext::Create(int device_ordinal,
                                                 CUdevice device) {
  CudaContext* context = nullptr;

  int flags = GetFlagsFromEnv();

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                 &former_primary_context_is_active)));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuDevicePrimaryCtxSetFlags(device, flags)));
    }
  }

  CUcontext former_context = CurrentContextOrDie();
  CUcontext new_context;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDevicePrimaryCtxRetain(&new_context, device)));
  if (former_context != nullptr) {
    CUdevice former_device;
    if (cuCtxGetDevice(&former_device) == CUDA_SUCCESS) {
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
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuCtxSetCurrent(former_context)));

  context = GetContextMap()->Add(new_context, device_ordinal);
  CHECK(context != nullptr)
      << "success in this call must entail non-null result";
  VLOG(2) << "created or reused context " << new_context << " for this thread";
  return context;
}

void CudaContext::SetActive() {
  TF_CHECK_OK(
      cuda::ToStatus(cuCtxSetCurrent(context_), "Failed setting context"));
}

bool CudaContext::IsActive() const { return CurrentContext() == context_; }

absl::Status CudaContext::Synchronize() {
  ScopedActivateContext activation(this);
  return cuda::ToStatus(cuCtxSynchronize());
}

}  // namespace stream_executor::gpu
