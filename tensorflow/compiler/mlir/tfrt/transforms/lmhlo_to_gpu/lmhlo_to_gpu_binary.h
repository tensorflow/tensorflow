// Copyright 2020 The TensorFlow Runtime Authors
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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"

namespace tensorflow {

struct GpuBinaryOptions {
  static GpuBinaryOptions DefaultGpuBinaryOptions() {
    GpuBinaryOptions options;
    options.platform_name = "CUDA";

    options.gpu_device_info.threads_per_block_limit = 1024;
    options.gpu_device_info.threads_per_warp = 32;
    options.gpu_device_info.shared_memory_per_block =
        49152;  // static shmem limit.
    // Should be 1024 for sm7.5, 1536 for sm8.6. This results in more blocks
    // than SMs on those architectures, but doesn't hit any resource limit.
    options.gpu_device_info.threads_per_core_limit = 2048;
    // This is higher than any SKU, resulting in more blocks than SMs.
    options.gpu_device_info.core_count = 128;
    options.gpu_device_info.block_dim_limit_x = 2147483647;
    options.gpu_device_info.block_dim_limit_y = 65535;
    options.gpu_device_info.block_dim_limit_z = 65535;

    options.cuda_compute_capability = {5, 2};
    options.rocm_compute_capability =
        stream_executor::RocmComputeCapability("gfx900");
    return options;
  }

  std::string platform_name;
  xla::gpu::GpuDeviceInfo gpu_device_info;
  stream_executor::CudaComputeCapability cuda_compute_capability;
  stream_executor::RocmComputeCapability rocm_compute_capability{"unknown"};
};

// Creates a pass that lowers lmhlo.fusion ops to a gpu.module with a binary
// device code attribute plus a gpu.launch_func.
std::unique_ptr<mlir::Pass> createConvertLmhloToGpuBinaryPass(
    GpuBinaryOptions options = GpuBinaryOptions::DefaultGpuBinaryOptions());

void registerConvertLmhloToGpuBinaryPass();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_
