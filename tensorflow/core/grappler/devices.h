/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_DEVICES_H_
#define TENSORFLOW_CORE_GRAPPLER_DEVICES_H_

#include <functional>
#include <utility>

#include "absl/types/variant.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// GpuVersion is used to abstract Gpu hardware version. On Cuda platform,
// it comprises a pair of integers denoting major and minor version.
// On ROCm platform, it comprises one integer for AMD GCN ISA version.
using GpuVersion = absl::variant<std::pair<int, int>, int>;

// Get the number of available GPUs.
// On CUDA platform, look for GPUs whose number of multiprocessors is no less
// than 8 and whose CUDA compute capability is no less than min_gpu_version,
// represented as a pair of integers.
// On ROCm platform, look for GPUs whose ISA version number is no less than
// min_gpu_version, represented as a single integer.
int GetNumAvailableGPUs(
#if GOOGLE_CUDA
    const GpuVersion& min_gpu_version = std::pair<int, int>(0, 0)
#elif TENSORFLOW_USE_ROCM
    const GpuVersion& min_gpu_version = 0
#endif
);

// Maximum amount of gpu memory available per gpu. gpu_id must be in the range
// [0, num_available_gpu)
int64 AvailableGPUMemory(int gpu_id);

// Get the number of logical CPU cores (aka hyperthreads) available.
int GetNumAvailableLogicalCPUCores();

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_DEVICES_H_
