/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_COMMON_CREATE_PJRT_CLIENT_UTIL_H_
#define TENSORFLOW_CORE_TFRT_COMMON_CREATE_PJRT_CLIENT_UTIL_H_

#include <memory>
#include <optional>
#include <set>

#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// Gets PJRT client from TFGlobalResourceManager. If it is not found, creates a
// PJRT client and adds it to TFGlobalResourceManager. Different `DeviceType`
// can choose to create the PJRT client explicitly (e.g. in ops) and add it to
// TFGlobalResourceManager, or create a PJRT client on the first use implicitly
// in this method.
// The inputs are the device_type of the caller, and an optional
// set of device IDs `allowed_devices` for which the stream executor will be
// created. `allowed_devices` is only used for GPU.
// TODO(b/260802979): consider passing `XlaPlatformInfo` for the options to
// create a client, or creating a class similar to `LocalClientOptions`.
// TODO(b/280111106): make PjrtClientFactoryOptions an input of
// GetOrCreatePjRtClient.
StatusOr<xla::PjRtClient*> GetOrCreatePjRtClient(
    const DeviceType& device_type,
    std::optional<std::set<int>> allowed_devices = std::nullopt);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_CREATE_PJRT_CLIENT_UTIL_H_
