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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TPU_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TPU_CLIENT_H_

#include <array>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/tpu/tpu_topology.h"

namespace xla {

class PjRtTpuDevice : public PjRtStreamExecutorDevice {
 public:
  PjRtTpuDevice(const tensorflow::tpu::TpuCoreLocationExternal core,
                std::unique_ptr<LocalDeviceState> local_device_state,
                int process_index, const std::array<int, 3>& coords,
                std::string device_kind)
      : PjRtStreamExecutorDevice(core.Id(), std::move(local_device_state),
                                 std::move(device_kind), process_index),
        core_(core),
        coords_(coords) {}

  const std::array<int, 3>& coords() const { return coords_; }
  int core_on_chip() const { return core_.index(); }
  const tensorflow::tpu::TpuCoreLocationExternal core() const { return core_; }

  std::string DebugString() const override {
    return absl::StrFormat("TPU_%i(process=%i,(%i,%i,%i,%i))", id(),
                           process_index(), coords_[0], coords_[1], coords_[2],
                           core_.index());
  }

 private:
  const tensorflow::tpu::TpuCoreLocationExternal core_;
  const std::array<int, 3> coords_;
};

class PjRtTpuClient : public PjRtStreamExecutorClient {
 public:
  PjRtTpuClient(LocalClient* client,
                std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
                int process_index);

  absl::string_view platform_version() const override {
    return platform_version_;
  }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  bool EnqueueD2DTransfersOnSrcStream() const override { return false; }

  StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const override;

  StatusOr<std::string> SerializeExecutable(
      const PjRtExecutable& executable) const override;

  StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized, std::unique_ptr<HloModule> hlo_module,
      CompileOptions options) override;

 private:
  const std::string platform_version_;
};

StatusOr<std::shared_ptr<PjRtClient>> GetTpuClient(
    int max_inflight_computations,
    absl::Duration init_retry_timeout = absl::ZeroDuration());

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TPU_CLIENT_H_
