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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_

#include <memory>
#include <optional>
#include <string>

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class StreamExecutorGpuDevice : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorGpuDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state,
                          std::string device_kind, std::string device_vendor,
                          int node_id, int slice_index = 0);

  int slice_index() const;

  absl::string_view device_vendor() const;

 private:
  std::string device_vendor_;
  int slice_index_;
};

// distributed_client may be nullptr in non-distributed settings.
// distributed_client should be in the connected state before calling this
// function.
StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id,
    const std::optional<std::set<int>>& allowed_devices = std::nullopt,
    std::optional<std::string> platform_name = std::nullopt,
    bool should_stage_host_to_device_transfers = true);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
