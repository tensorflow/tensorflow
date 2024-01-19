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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/service_executable_run_options.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

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

//===----------------------------------------------------------------------===//
// NcclExecuteParams
//===----------------------------------------------------------------------===//

using GlobalDeviceIdMap = NcclExecuteParams::GlobalDeviceIdMap;

// Returns global device id for a local device ordinal or an error if global
// device id map is misconfigured and missing an entry for a local device.
static absl::StatusOr<GlobalDeviceId> GetGlobalDeviceId(
    const GlobalDeviceIdMap* device_id_map, int64_t local_device_ordinal) {
  // No local -> global mapping was provided; assume the identity mapping.
  if (!device_id_map) return GlobalDeviceId(local_device_ordinal);

  // Find a global device id in a global device id map.
  auto it = device_id_map->find(local_device_ordinal);
  if (it == device_id_map->end())
    return absl::NotFoundError(
        absl::StrCat("No global device id found for local device ordinal: ",
                     local_device_ordinal));

  return it->second;
}

absl::StatusOr<NcclExecuteParams> NcclExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options,
    int64_t local_device_ordinal) {
  const GpuExecutableRunOptions* gpu_options =
      run_options.run_options().gpu_executable_run_options();

  auto* device_id_map = gpu_options && gpu_options->gpu_global_device_ids()
                            ? &*gpu_options->gpu_global_device_ids()
                            : nullptr;

  auto* nccl_callback = gpu_options && gpu_options->nccl_clique_id_callback()
                            ? &gpu_options->nccl_clique_id_callback()
                            : nullptr;

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      GetGlobalDeviceId(device_id_map, local_device_ordinal));

  return NcclExecuteParams(run_options.run_options().run_id(),
                           local_device_ordinal, global_device_id,
                           run_options.run_options().device_assignment(),
                           device_id_map, nccl_callback);
}

NcclExecuteParams::NcclExecuteParams(
    RunId run_id, int64_t local_device_ordinal, GlobalDeviceId global_device_id,
    const DeviceAssignment* device_assn,
    const GlobalDeviceIdMap* global_device_id_map,
    const NcclCliqueIdCallback* nccl_clique_id_callback)
    : run_id_(run_id),
      local_device_ordinal_(local_device_ordinal),
      global_device_id_(global_device_id),
      device_assn_(device_assn),
      global_device_id_map_(global_device_id_map),
      nccl_clique_id_callback_(nccl_clique_id_callback) {}

RunId NcclExecuteParams::run_id() const { return run_id_; }

int64_t NcclExecuteParams::local_device_ordinal() const {
  return local_device_ordinal_;
}

GlobalDeviceId NcclExecuteParams::global_device_id() const {
  return global_device_id_;
}

const GlobalDeviceIdMap* NcclExecuteParams::global_device_id_map() const {
  return global_device_id_map_;
}

const DeviceAssignment* NcclExecuteParams::device_assn() const {
  return device_assn_;
}

const NcclCliqueIdCallback* NcclExecuteParams::nccl_clique_id_callback() const {
  return nccl_clique_id_callback_;
}

}  // namespace gpu
}  // namespace xla
