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

#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_platform_id.h"

namespace tensorflow {
namespace tpu {

template <typename T>
using StatusOr = TpuComputationPlacer::StatusOr<T>;

TpuComputationPlacer::TpuComputationPlacer() {
  placer_ = stream_executor::tpu::ExecutorApiFn()->TpuComputationPlacer_NewFn();
}

TpuComputationPlacer::~TpuComputationPlacer() {
  stream_executor::tpu::ExecutorApiFn()->TpuComputationPlacer_FreeFn(placer_);
}

StatusOr<int> TpuComputationPlacer::DeviceId(int replica, int computation,
                                             int replica_count,
                                             int computation_count) {
  LOG(FATAL) << "Unimplemented.";
}

StatusOr<xla::DeviceAssignment> TpuComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
  StatusHelper status;
  xla::DeviceAssignment result(replica_count, computation_count);
  xla::Array2D<int> result_int32(replica_count, computation_count);
  stream_executor::tpu::ExecutorApiFn()->TpuComputationPlacer_AssignDevicesFn(
      placer_, replica_count, computation_count, result_int32.data(),
      status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  // Upcast to 64-bit.
  for (int i = 0; i < replica_count; ++i) {
    for (int j = 0; j < computation_count; ++j)
      result(i, j) = result_int32(i, j);
  }
  return result;
}

/*static*/ StatusOr<xla::DeviceAssignment>
TpuComputationPlacer::AssignLocalDevices(TpuHostLocationExternal host_location,
                                         int replica_count,
                                         int computation_count) {
  StatusHelper status;
  xla::DeviceAssignment result(replica_count, computation_count);
  xla::Array2D<int> result_int32(replica_count, computation_count);
  stream_executor::tpu::ExecutorApiFn()
      ->TpuComputationPlacer_AssignLocalDevicesFn(
          host_location.impl(), replica_count, computation_count,
          result_int32.data(), status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  // Upcast to 64-bit.
  for (int i = 0; i < replica_count; ++i) {
    for (int j = 0; j < computation_count; ++j)
      result(i, j) = result_int32(i, j);
  }
  return result;
}

static std::unique_ptr<xla::ComputationPlacer> CreateTpuComputationPlacer() {
  return std::make_unique<TpuComputationPlacer>();
}

static bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(GetTpuPlatformId(),
                                                    CreateTpuComputationPlacer);
  return true;
}
static bool module_initialized = InitModule();

}  // namespace tpu
}  // namespace tensorflow
