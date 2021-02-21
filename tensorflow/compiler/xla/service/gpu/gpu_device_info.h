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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_

namespace xla {
namespace gpu {

// THe information contained in these structures is also contained in
// se::DeviceDescription, but separating these out lets us write code that does
// not depend on stream executor.

struct CudaComputeCapability {
  int cc_major;
  int cc_minor;
};

struct GpuDeviceInfo {
  int threads_per_block_limit;
  int threads_per_warp;
  int shared_memory_per_block;
  int threads_per_core_limit;
  int core_count;
  int block_dim_limit_x;
  int block_dim_limit_y;
  int block_dim_limit_z;
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_
