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

#ifndef XLA_SERVICE_TPU_COMPUTATION_PLACER_H_
#define XLA_SERVICE_TPU_COMPUTATION_PLACER_H_

#include "xla/service/computation_placer.h"
#include "xla/statusor.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "xla/stream_executor/tpu/tpu_topology.h"

namespace tensorflow {
namespace tpu {

class TpuComputationPlacer : public xla::ComputationPlacer {
 public:
  template <typename T>
  using StatusOr = absl::StatusOr<T>;

  TpuComputationPlacer();
  ~TpuComputationPlacer() override;

  StatusOr<int> DeviceId(int replica, int computation, int replica_count,
                         int computation_count) override;

  StatusOr<xla::DeviceAssignment> AssignDevices(int replica_count,
                                                int computation_count) override;

  static StatusOr<xla::DeviceAssignment> AssignLocalDevices(
      TpuHostLocationExternal host_location, int replica_count,
      int computation_count);

 private:
  XLA_ComputationPlacer* placer_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_SERVICE_TPU_COMPUTATION_PLACER_H_
