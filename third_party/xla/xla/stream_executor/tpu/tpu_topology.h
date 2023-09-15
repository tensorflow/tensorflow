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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_TOPOLOGY_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_TOPOLOGY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xla/stream_executor/tpu/c_api_decl.h"
#include "tsl/platform/types.h"

namespace tensorflow {
namespace tpu {

struct TpuDimensionsExternal {
  int x;
  int y;
  int z;
};

class TpuCoreLocationExternal {
 public:
  TpuCoreLocationExternal() : core_location_(nullptr) {}
  explicit TpuCoreLocationExternal(SE_TpuTopology_Core* core_location)
      : core_location_(core_location) {}
  TpuDimensionsExternal chip_coordinates() const;
  TpuDimensionsExternal host_coordinates() const;
  int32_t index() const;
  int32_t Id() const;

  SE_TpuTopology_Core* impl() const { return core_location_; }

 private:
  SE_TpuTopology_Core* core_location_;
};

class TpuHostLocationExternal {
 public:
  explicit TpuHostLocationExternal(SE_TpuTopology_Host* host_location)
      : host_location_(host_location) {}
  int32_t Id() const;
  std::vector<TpuCoreLocationExternal> Cores(TpuCoreTypeEnum core_type) const;

  SE_TpuTopology_Host* impl() const { return host_location_; }

 private:
  SE_TpuTopology_Host* host_location_;
};

struct TpuTopologyChipBoundsExternal {
  int x;
  int y;
  int z;
};

class TpuTopologyExternal {
 public:
  explicit TpuTopologyExternal(SE_TpuTopology* topology)
      : topology_(topology) {}
  int32_t LogicalDevicesPerHost(TpuCoreTypeEnum core_type) const;
  int32_t LogicalDevicesPerChip(TpuCoreTypeEnum core_type) const;
  int32_t HostCount() const;
  int32_t ChipsPerHost() const;
  TpuTopologyChipBoundsExternal chip_bounds() const;
  bool HasChip(int x, int y, int z) const;
  TpuCoreLocationExternal CoreForId(TpuCoreTypeEnum core_type, int id) const;
  TpuCoreLocationExternal Core(TpuCoreTypeEnum core_type, int x, int y, int z,
                               int index) const;
  std::vector<TpuCoreLocationExternal> cores(TpuCoreTypeEnum core_type) const;
  int IdForHost(TpuDimensionsExternal host) const;
  TpuVersionEnum version() const;

 private:
  SE_TpuTopology* topology_;
};

std::string TpuVersionEnumToString(TpuVersionEnum version);

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_TOPOLOGY_H_
