/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"

#include "absl/algorithm/container.h"

namespace xla {
namespace gpu {

NcclCliqueKey::NcclCliqueKey(std::vector<GlobalDeviceId> devices)
    : devices_(std::move(devices)) {}

std::string NcclCliqueKey::ToString() const {
  return GlobalDeviceIdsToString(devices_);
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_gpu_global_device_ids(
    absl::optional<std::vector<GlobalDeviceId>> gpu_global_device_ids) {
  gpu_global_device_ids_ = std::move(gpu_global_device_ids);
  return *this;
}

const absl::optional<std::vector<GlobalDeviceId>>&
GpuExecutableRunOptions::gpu_global_device_ids() const {
  return gpu_global_device_ids_;
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_nccl_unique_id_callback(
    NcclUniqueIdCallback nccl_unique_id_callback) {
  nccl_unique_id_callback_ = std::move(nccl_unique_id_callback);
  return *this;
}

const NcclUniqueIdCallback& GpuExecutableRunOptions::nccl_unique_id_callback()
    const {
  return nccl_unique_id_callback_;
}

}  // namespace gpu
}  // namespace xla
