/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/green_context.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"

namespace stream_executor::gpu {

GreenContext::GreenContext(CUdevice device, CUgreenCtx green_ctx, int sm_count)
    : device_(device), green_ctx_(green_ctx), sm_count_(sm_count) {}

GreenContext::~GreenContext() {
  if (green_ctx_ != nullptr) {
    absl::Status status = cuda::ToStatus(cuGreenCtxDestroy(green_ctx_));
    if (!status.ok()) {
      LOG(ERROR) << "Failed to destroy green context: " << status;
    }
  }
}

absl::StatusOr<std::unique_ptr<GreenContext>> GreenContext::Create(
    CUdevice device, CUdevResource sm_resource) {
  if (sm_resource.type != CU_DEV_RESOURCE_TYPE_SM) {
    return absl::InvalidArgumentError(
        "GreenContext::Create requires a CU_DEV_RESOURCE_TYPE_SM resource");
  }

  // cuDevResourceGenerateDesc takes a non-const resource pointer, so operate on
  // the local copy.
  CUdevResourceDesc desc;
  ABSL_RETURN_IF_ERROR(
      cuda::ToStatus(cuDevResourceGenerateDesc(&desc, &sm_resource, 1)));

  CUgreenCtx green_ctx = nullptr;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(
      cuGreenCtxCreate(&green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM)));

  return absl::WrapUnique(new GreenContext(
      device, green_ctx, static_cast<int>(sm_resource.sm.smCount)));
}

absl::StatusOr<std::unique_ptr<GreenContext>> GreenContext::CreateWithSmCount(
    CUdevice device, int sm_count) {
  if (sm_count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid SM count for green context: ", sm_count));
  }

  CUdevResource device_sms;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetDevResource(device, &device_sms, CU_DEV_RESOURCE_TYPE_SM)));

  if (sm_count > static_cast<int>(device_sms.sm.smCount)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested ", sm_count, " SMs for green context but device only has ",
        device_sms.sm.smCount));
  }

  // Carve a single partition of (at least) `sm_count` SMs off the device's SM
  // resource. The remainder is discarded here; this factory exists to create a
  // single standalone green context.
  CUdevResource group;
  CUdevResource remaining;
  unsigned int num_groups = 1;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(cuDevSmResourceSplitByCount(
      &group, &num_groups, &device_sms, &remaining, /*useFlags=*/0,
      /*minCount=*/static_cast<unsigned int>(sm_count))));
  if (num_groups < 1) {
    return absl::InternalError(
        "cuDevSmResourceSplitByCount produced no SM groups");
  }

  return Create(device, group);
}

absl::StatusOr<CUstream> GreenContext::CreateStream(int priority) const {
  CUstream stream = nullptr;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(cuGreenCtxStreamCreate(
      &stream, green_ctx_, CU_STREAM_NON_BLOCKING, priority)));
  return stream;
}

}  // namespace stream_executor::gpu
