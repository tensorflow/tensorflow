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

#include "tensorflow/core/tfrt/gpu/gpu_shared_context.h"

#include <string>
#include <utility>
#include <vector>

namespace tfrt {
namespace gpu {

GpuSharedContext::GpuSharedContext(
    int64_t run_id,
    absl::flat_hash_map<LocalDeviceIdentifier, int> local_ids_to_rank,
    std::vector<int64_t> gpu_global_device_ids,
    XcclUniqueIdCallback xccl_unique_id_callback,
    const std::string* compiled_code)
    : run_id_(run_id),
      local_ids_to_rank_(std::move(local_ids_to_rank)),
      gpu_global_device_ids_(std::move(gpu_global_device_ids)),
      xccl_unique_id_callback_(std::move(xccl_unique_id_callback)),
      compiled_code_(compiled_code) {}

}  // namespace gpu
}  // namespace tfrt
