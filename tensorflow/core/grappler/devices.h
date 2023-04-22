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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// Get the number of available GPUs whose number of multiprocessors is no less
// than 8 and whose CUDA compute capability is no less than
// min_cuda_compute_capability.
int GetNumAvailableGPUs(
    const std::pair<int, int>& min_cuda_compute_capability = {0, 0});

// Maximum amount of gpu memory available per gpu. gpu_id must be in the range
// [0, num_available_gpu)
int64 AvailableGPUMemory(int gpu_id);

// Get the number of logical CPU cores (aka hyperthreads) available.
int GetNumAvailableLogicalCPUCores();

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_DEVICES_H_
