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

#include "xla/stream_executor/cuda/locality_domain.h"

#include <cstring>  // IWYU pragma: keep
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/green_context.h"

namespace stream_executor::gpu {

LocalityDomain::LocalityDomain(int locality_domain_id,
                               std::unique_ptr<GreenContext> green_context)
    : locality_domain_id_(locality_domain_id),
      green_context_(std::move(green_context)) {}

absl::StatusOr<int> GetLocalityDomainCount(CUdevice device) {
#if CUDA_VERSION >= 13040
  // cuDriverGetVersion uses the same encoding as CUDA_VERSION:
  // 1000 * major + 10 * minor. CUDA 13.4 is 13040.
  int driver_version = 0;
  absl::Status driver_status =
      cuda::ToStatus(cuDriverGetVersion(&driver_version),
                     "Failed to query CUDA driver version");
  if (!driver_status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to query CUDA driver version: ", driver_status.ToString()));
  }
  if (driver_version < 13040) {
    return absl::UnimplementedError(absl::StrCat(
        "CUDA locality domains require CUDA driver 13.4 or newer; driver "
        "version is ",
        driver_version));
  }

  int count = 0;
  absl::Status status = cuda::ToStatus(cuDeviceGetAttribute(
      &count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device));
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to query CUDA locality domain count: ", status.ToString()));
  }
  if (count <= 0) {
    return absl::InternalError(
        absl::StrCat("Invalid CUDA locality domain count for device: ", count));
  }
  return count;
#else
  (void)device;
  return absl::UnimplementedError(
      "CUDA locality domains require CUDA 13.4 or newer");
#endif  // CUDA_VERSION >= 13040
}

absl::StatusOr<std::vector<std::unique_ptr<LocalityDomain>>>
CreateLocalityDomains(CUdevice device) {
#if CUDA_VERSION >= 13040
  ABSL_ASSIGN_OR_RETURN(int count, GetLocalityDomainCount(device));

  CUdevResource sm_resource;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM)));

  const int total_sms = static_cast<int>(sm_resource.sm.smCount);
  const int sms_per_domain = total_sms / count;

  // Split the device's SMs into one group per locality domain, pinning each
  // group to its locality domain id and backfilling any remainder.
  std::vector<CUdevResource> domain_sms(count);
  std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(count);
  CUdevResource remainder;
  for (int i = 0; i < count; ++i) {
    std::memset(&params[i], 0, sizeof(params[i]));
    params[i].smCount = sms_per_domain;
    params[i].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID |
                      CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
    params[i].localityDomainId = i;
  }

  ABSL_RETURN_IF_ERROR(cuda::ToStatus(
      cuDevSmResourceSplit(domain_sms.data(), count, &sm_resource, &remainder,
                           /*flags=*/0, params.data())));

  std::vector<std::unique_ptr<LocalityDomain>> domains;
  domains.reserve(count);
  for (int i = 0; i < count; ++i) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<GreenContext> green_context,
                     GreenContext::Create(device, domain_sms[i]));
    domains.push_back(
        std::make_unique<LocalityDomain>(i, std::move(green_context)));
  }
  return domains;
#else
  (void)device;
  return absl::UnimplementedError(
      "CUDA locality domains require CUDA 13.4 or newer");
#endif  // CUDA_VERSION >= 13040
}

}  // namespace stream_executor::gpu
