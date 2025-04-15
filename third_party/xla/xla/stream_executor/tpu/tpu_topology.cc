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

#include "xla/stream_executor/tpu/tpu_topology.h"

#include <cstdint>
#include <string>
#include <vector>

#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"

namespace tensorflow {
namespace tpu {

TpuDimensionsExternal TpuCoreLocationExternal::chip_coordinates() const {
  int x, y, z;
  stream_executor::tpu::ExecutorApiFn()->TpuCoreLocation_ChipCoordinatesFn(
      core_location_, &x, &y, &z);
  return {x, y, z};
}

TpuDimensionsExternal TpuCoreLocationExternal::host_coordinates() const {
  int x, y, z;
  stream_executor::tpu::ExecutorApiFn()->TpuCoreLocation_HostCoordinatesFn(
      core_location_, &x, &y, &z);
  return {x, y, z};
}

int32_t TpuCoreLocationExternal::index() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuCoreLocation_IndexFn(
      core_location_);
}

int32_t TpuCoreLocationExternal::Id() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuCoreLocation_IdFn(
      core_location_);
}

int32_t TpuHostLocationExternal::Id() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuHostLocation_IdFn(
      host_location_);
}

std::vector<TpuCoreLocationExternal> TpuHostLocationExternal::Cores(
    TpuCoreTypeEnum core_type) const {
  int num_cores =
      stream_executor::tpu::ExecutorApiFn()->TpuHostLocation_NumCoresFn(
          host_location_, core_type);
  std::vector<SE_TpuTopology_Core*> core_ptrs(num_cores);
  stream_executor::tpu::ExecutorApiFn()->TpuHostLocation_CoresFn(
      host_location_, core_type, core_ptrs.data());
  std::vector<TpuCoreLocationExternal> result;
  result.reserve(num_cores);
  for (SE_TpuTopology_Core* ptr : core_ptrs) {
    result.emplace_back(ptr);
  }
  return result;
}

int32_t TpuTopologyExternal::LogicalDevicesPerHost(
    TpuCoreTypeEnum core_type) const {
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuTopology_LogicalDevicesPerHostFn(topology_, core_type);
}

int32_t TpuTopologyExternal::LogicalDevicesPerChip(
    TpuCoreTypeEnum core_type) const {
  return stream_executor::tpu::ExecutorApiFn()
      ->TpuTopology_LogicalDevicesPerChipFn(topology_, core_type);
}

int32_t TpuTopologyExternal::HostCount() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuTopology_HostCountFn(
      topology_);
}

int32_t TpuTopologyExternal::ChipsPerHost() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuTopology_ChipsPerHostFn(
      topology_);
}

TpuTopologyChipBoundsExternal TpuTopologyExternal::chip_bounds() const {
  return {stream_executor::tpu::ExecutorApiFn()->TpuTopology_ChipBounds_XFn(
              topology_),
          stream_executor::tpu::ExecutorApiFn()->TpuTopology_ChipBounds_YFn(
              topology_),
          stream_executor::tpu::ExecutorApiFn()->TpuTopology_ChipBounds_ZFn(
              topology_)};
}

bool TpuTopologyExternal::HasChip(int x, int y, int z) const {
  return stream_executor::tpu::ExecutorApiFn()->TpuTopology_HasChipFn(topology_,
                                                                      x, y, z);
}

TpuCoreLocationExternal TpuTopologyExternal::CoreForId(
    TpuCoreTypeEnum core_type, int id) const {
  return TpuCoreLocationExternal(
      stream_executor::tpu::ExecutorApiFn()->TpuTopology_CoreForIdFn(
          topology_, core_type, id));
}

TpuCoreLocationExternal TpuTopologyExternal::Core(TpuCoreTypeEnum core_type,
                                                  int x, int y, int z,
                                                  int index) const {
  return TpuCoreLocationExternal(
      stream_executor::tpu::ExecutorApiFn()->TpuTopology_CoreFn(
          topology_, core_type, x, y, z, index));
}

std::vector<TpuCoreLocationExternal> TpuTopologyExternal::cores(
    TpuCoreTypeEnum core_type) const {
  int num_cores = stream_executor::tpu::ExecutorApiFn()->TpuTopology_NumCoresFn(
      topology_, core_type);
  std::vector<SE_TpuTopology_Core*> core_ptrs(num_cores);
  stream_executor::tpu::ExecutorApiFn()->TpuTopology_CoresFn(
      topology_, core_type, core_ptrs.data());
  std::vector<TpuCoreLocationExternal> result;
  result.reserve(num_cores);
  for (SE_TpuTopology_Core* ptr : core_ptrs) {
    result.emplace_back(ptr);
  }
  return result;
}

int TpuTopologyExternal::IdForHost(TpuDimensionsExternal host) const {
  return stream_executor::tpu::ExecutorApiFn()->TpuTopology_IdForHostFn(
      topology_, host.x, host.y, host.z);
}

TpuVersionEnum TpuTopologyExternal::version() const {
  return stream_executor::tpu::ExecutorApiFn()->TpuTopology_VersionFn(
      topology_);
}

std::string TpuVersionEnumToString(TpuVersionEnum version) {
  switch (version) {
    case kUnknownTpuVersion:
      return "Unknown TPU version";
    case kTpuV2:
      return "TPU v2";
    case kTpuV3:
      return "TPU v3";
    case kTpuV4:
      return "TPU v4";
    case kTpuV5:
      return "TPU v5";
  }
}

}  // namespace tpu
}  // namespace tensorflow
