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

#include "xla/service/tpu_computation_placer.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/array2d.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_platform_id.h"
#include "xla/stream_executor/tpu/tpu_topology.h"

namespace tensorflow {
namespace tpu {
using stream_executor::tpu::ExecutorApiFn;

TpuComputationPlacer::TpuComputationPlacer() {
  placer_ = ExecutorApiFn()->TpuComputationPlacer_NewFn();
}

TpuComputationPlacer::~TpuComputationPlacer() {
  ExecutorApiFn()->TpuComputationPlacer_FreeFn(placer_);
}

namespace {
absl::StatusOr<xla::DeviceAssignment> ToAssignment(
    int replica_count, int computation_count,
    const xla::Array2D<int>& result_int32, const StatusHelper& status) {
  if (!status.ok()) {
    return status.status();
  }
  xla::DeviceAssignment result(replica_count, computation_count);
  // Upcast to 64-bit.
  for (int i = 0; i < replica_count; ++i) {
    for (int j = 0; j < computation_count; ++j) {
      result(i, j) = result_int32(i, j);
    }
  }
  return result;
}
}  // namespace

absl::StatusOr<xla::DeviceAssignment> TpuComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
  StatusHelper status;
  xla::Array2D<int> result_int32(replica_count, computation_count);
  ExecutorApiFn()->TpuComputationPlacer_AssignDevicesFn(
      placer_, replica_count, computation_count, result_int32.data(),
      status.c_status);
  return ToAssignment(replica_count, computation_count, result_int32, status);
}

/*static*/ absl::StatusOr<xla::DeviceAssignment>
TpuComputationPlacer::AssignLocalDevices(TpuHostLocationExternal host_location,
                                         int replica_count,
                                         int computation_count) {
  StatusHelper status;
  xla::Array2D<int> result_int32(replica_count, computation_count);
  ExecutorApiFn()->TpuComputationPlacer_AssignLocalDevicesFn(
      host_location.impl(), replica_count, computation_count,
      result_int32.data(), status.c_status);
  return ToAssignment(replica_count, computation_count, result_int32, status);
}

namespace {
std::unique_ptr<xla::ComputationPlacer> CreateTpuComputationPlacer() {
  return std::make_unique<TpuComputationPlacer>();
}

bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(GetTpuPlatformId(),
                                                    CreateTpuComputationPlacer);
  return true;
}
bool module_initialized = InitModule();
}  //  namespace

}  // namespace tpu
}  // namespace tensorflow
