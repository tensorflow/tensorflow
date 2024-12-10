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

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_options.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tsl/platform/statusor.h"
namespace xla {

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetGpuClient(
    const PjrtClientFactoryOptions& option) {
  xla::GpuClientOptions gpu_client_options;
  gpu_client_options.node_id = option.gpu_options.node_id;
  gpu_client_options.num_nodes = 1;
  gpu_client_options.allowed_devices = option.gpu_options.allowed_devices;
  gpu_client_options.platform_name = option.gpu_options.platform_name;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      xla::GetXlaPjrtGpuClient(gpu_client_options));
  return std::move(client);
}

REGISTER_PJRT_CLIENT_FACTORY(gpu_client, tensorflow::DEVICE_GPU, GetGpuClient);

REGISTER_PJRT_CLIENT_FACTORY(xla_gpu_client, tensorflow::DEVICE_XLA_GPU,
                             GetGpuClient);

}  // namespace xla
