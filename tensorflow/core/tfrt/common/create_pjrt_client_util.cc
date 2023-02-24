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
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"

#include <memory>
#include <optional>
#include <set>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"

namespace tensorflow {

StatusOr<xla::PjRtClient*> GetOrCreatePjRtClient(
    const DeviceType& device_type,
    std::optional<std::set<int>> allowed_devices) {
  StatusOr<xla::PjRtClient*> existing_pjrt_client = GetPjRtClient(device_type);
  // Checks whether a PJRT client is found first as the DeviceType can choose to
  // create the PJRT client explicitly (e.g. in ops).
  if (existing_pjrt_client.ok()) {
    return *existing_pjrt_client;
  }
  // Returns directly if the error is not NotFound.
  if (!tsl::errors::IsNotFound(existing_pjrt_client.status())) {
    return existing_pjrt_client;
  }
  // TODO(b/260799193): use XlaPlatformInfo to pass device-specific options.
  // This info should be set in the plugin init for next pluggable device.
  if (device_type != DEVICE_XLA_GPU) {
    return errors::Unimplemented(
        "The PJRT client for ", device_type,
        " is not created explicitly before its first use and creating this "
        "PJRT client on the first use is not implemented.");
  }
  xla::GpuAllocatorConfig allocator_config;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> pjrt_client,
                      xla::GetStreamExecutorGpuClient(
                          /*asynchronous=*/true, allocator_config,
                          /*distributed_client=*/nullptr,
                          /*node_id=*/0, allowed_devices));
  // Gets a pointer of pjrt_client because the ownership of pjrt_client will be
  // transferred in the SetPjRtClientInTFGlobalResourceManager call below.
  auto pjrt_client_ptr = pjrt_client.get();
  TF_RETURN_IF_ERROR(SetPjRtClientInTFGlobalResourceManager(
      device_type, std::move(pjrt_client)));
  return pjrt_client_ptr;
}

}  // namespace tensorflow
