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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// Key for naming up a particular NCCL clique.  This is just a set of unique
// device IDs (i.e. GPU IDs). The device IDs must be global within a cluster.
class NcclCliqueKey {
 public:
  explicit NcclCliqueKey(std::vector<GlobalDeviceId> devices);

  template <typename H>
  friend H AbslHashValue(H h, const NcclCliqueKey& k) {
    return H::combine(std::move(h), k.devices_);
  }
  friend bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b) {
    return a.devices_ == b.devices_;
  }

  const std::vector<GlobalDeviceId>& devices() const { return devices_; }

  std::string ToString() const;

 private:
  std::vector<GlobalDeviceId> devices_;
};

using NcclUniqueIdCallback =
    std::function<StatusOr<std::string>(const NcclCliqueKey&)>;

// GPU-specific executable options.
// We keep these separate from ExecutableRunOptions to avoid adding
// dependencies to ExecutableRunOptions.
class GpuExecutableRunOptions {
 public:
  // Sets a mapping from local device ordinals to global device IDs.
  // Used only on NVidia GPUs for cross-host NCCL collectives. If set, the
  // elements of `device_assignment` are interpreted as global device IDs, not
  // local device ordinals.
  GpuExecutableRunOptions& set_gpu_global_device_ids(
      std::optional<std::vector<GlobalDeviceId>> gpu_global_device_ids);
  const std::optional<std::vector<GlobalDeviceId>>& gpu_global_device_ids()
      const;

  // Callback that returns a ncclUniqueId encoded as a string for a group of
  // communicating GPU devices. Used only on NVidia GPUs.
  GpuExecutableRunOptions& set_nccl_unique_id_callback(
      NcclUniqueIdCallback nccl_unique_id_callback);
  const NcclUniqueIdCallback& nccl_unique_id_callback() const;

 private:
  std::optional<std::vector<GlobalDeviceId>> gpu_global_device_ids_;
  NcclUniqueIdCallback nccl_unique_id_callback_;
};

// NCCL-related execution parameters.
struct NcclExecuteParams {
  NcclExecuteParams(const ServiceExecutableRunOptions& run_options,
                    se::Stream* stream);

  se::Stream* stream;
  RunId run_id;
  const DeviceAssignment* device_assn;                       // never null
  const std::vector<GlobalDeviceId>* gpu_global_device_ids;  // may be null
  const NcclUniqueIdCallback* nccl_unique_id_callback;       // may be null

  StatusOr<GlobalDeviceId> GetGlobalDeviceId() const;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
