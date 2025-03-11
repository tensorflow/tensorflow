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

#include <cstdint>
#include <map>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"

namespace xla::gpu {

GpuExecutableRunOptions& GpuExecutableRunOptions::set_gpu_global_device_ids(
    std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids) {
  gpu_global_device_ids_ = std::move(gpu_global_device_ids);
  return *this;
}

const std::optional<std::map<int, GlobalDeviceId>>&
GpuExecutableRunOptions::gpu_global_device_ids() const {
  return gpu_global_device_ids_;
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_clique_id_callback(
    CliqueIdCallback clique_id_callback) {
  clique_id_callback_ = std::move(clique_id_callback);
  return *this;
}

const CliqueIdCallback& GpuExecutableRunOptions::clique_id_callback() const {
  return clique_id_callback_;
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_collectives(
    GpuCollectives* collectives) {
  collectives_ = collectives;
  return *this;
}

GpuCollectives* GpuExecutableRunOptions::collectives() const {
  return collectives_;
}

GpuExecutableRunOptions& GpuExecutableRunOptions::set_incarnations(
    absl::flat_hash_map<GlobalDeviceId, uint64_t> incarnations) {
  incarnations_ = std::move(incarnations);
  return *this;
}

const std::optional<absl::flat_hash_map<GlobalDeviceId, uint64_t>>&
GpuExecutableRunOptions::incarnations() const {
  return incarnations_;
}

}  // namespace xla::gpu
