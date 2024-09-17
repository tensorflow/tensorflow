/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_COMMON_PJRT_UTIL_H_
#define TENSORFLOW_CORE_TFRT_COMMON_PJRT_UTIL_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/pjrt_state.h"

namespace tensorflow {

// Sets PJRT client for device_type in TFGlobalResourceManager. If a PJRT client
// for this device_type already exists, the existing PJRT client will not be
// destroyed, and will be kept alive in an "unused client" vector. PJRT API
// semantics require the PJRT client to outlive PJRT buffers.
Status SetPjRtClientInTFGlobalResourceManager(
    const DeviceType& device_type, std::unique_ptr<xla::PjRtClient> client);

// Gets (the most recent) PJRT client for device_type from
// TFGlobalResourceManager.
absl::StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type);

Status SetPjRtGpuClientCreationInfoInTFGlobalResourceManager(
    std::unique_ptr<PjRtGpuClientCreationInfo> info);
absl::StatusOr<PjRtGpuClientCreationInfo*> GetPjRtGpuClientCreationInfo();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_PJRT_UTIL_H_
