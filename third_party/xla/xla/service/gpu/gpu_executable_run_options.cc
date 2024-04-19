/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_executable_run_options.h"

#include <map>
#include <optional>
#include <utility>

#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique_key.h"

namespace xla {
namespace gpu {

GpuExecutableRunOptions& GpuExecutableRunOptions::set_gpu_global_device_ids(
    std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids) {
  gpu_global_device_ids_ = std::move(gpu_global_device_ids);
  return *this;
}

const std::optional<std::map<int, GlobalDeviceId>>&
GpuExecutableRunOptions::gpu_global_device_ids() const {
  return gpu_global_device_ids_;
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_nccl_clique_id_callback(
    NcclCliqueIdCallback nccl_clique_id_callback) {
  nccl_clique_id_callback_ = std::move(nccl_clique_id_callback);
  return *this;
}

const NcclCliqueIdCallback& GpuExecutableRunOptions::nccl_clique_id_callback()
    const {
  return nccl_clique_id_callback_;
}

}  // namespace gpu
}  // namespace xla
