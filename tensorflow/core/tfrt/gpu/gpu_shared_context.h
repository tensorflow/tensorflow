// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_CORE_TFRT_GPU_GPU_SHARED_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_GPU_GPU_SHARED_CONTEXT_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace gpu {

// Key for naming up a particular XCCL clique.  This is just a set of unique
// device IDs (i.e. GPU IDs). The device IDs must be global within a collective.
using XcclCliqueKey = std::vector<int64_t>;

// Callback that returns a ncclUniqueId encoded as a string for a group of
// communicating GPU devices.
using XcclUniqueIdCallback =
    std::function<Expected<std::string>(const XcclCliqueKey&)>;

// TODO(hanbinyoon): Rename this class appropriately.
// This class contains stateful resources needed to compile and execute programs
// in the XLA GPU integration environment.
class GpuSharedContext {
 public:
  // For BefThunk integration, this is the device ordinal.
  typedef int LocalDeviceIdentifier;

  explicit GpuSharedContext(
      int64_t run_id,
      absl::flat_hash_map<LocalDeviceIdentifier, int> local_ids_to_rank,
      std::vector<int64_t> gpu_global_device_ids,
      XcclUniqueIdCallback xccl_unique_id_callback,
      const std::string* compiled_code);

  // Accessors
  int64_t run_id() const { return run_id_; }
  const absl::flat_hash_map<LocalDeviceIdentifier, int>& local_ids_to_rank()
      const {
    return local_ids_to_rank_;
  }
  const std::vector<int64_t>& gpu_global_device_ids() const {
    return gpu_global_device_ids_;
  }
  const XcclUniqueIdCallback& xccl_unique_id_callback() const {
    return xccl_unique_id_callback_;
  }
  const std::string* compiled_code() const { return compiled_code_; }

 private:
  int64_t run_id_;
  const absl::flat_hash_map<LocalDeviceIdentifier, int> local_ids_to_rank_;
  const std::vector<int64_t>& gpu_global_device_ids_;
  const XcclUniqueIdCallback xccl_unique_id_callback_;

  // The compiled code is PTX in Cuda and unused empty string in ROCm.
  const std::string* compiled_code_;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_GPU_GPU_SHARED_CONTEXT_H_
