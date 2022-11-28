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

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

Status SetPjRtClientInTFGlobalResourceManager(
    const DeviceType& device_type, std::unique_ptr<xla::PjRtClient> client);

// Attempt to delete PJRT client from TFGlobalResourceManager. Returns OK if the
// deletion succeeded, or if the PJRT resource was not found. Else return the
// deletion error.
Status DeletePjRtClientFromTFGlobalResourceManagerIfResourceExists(
    const DeviceType& device_type);

StatusOr<xla::PjRtClient*> GetPjRtClientFromTFGlobalResourceManager(
    const DeviceType& device_type);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_PJRT_UTIL_H_
