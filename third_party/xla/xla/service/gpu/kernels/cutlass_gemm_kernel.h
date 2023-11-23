/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNEL_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNEL_H_

#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel {

// A reference implementation GEMM kernel written in CUTLASS based on
// `00_basic_gemm` example.
StatusOr<CustomKernel> GetCutlassGemmKernel(PrimitiveType dtype, int32_t m,
                                            int32_t n, int32_t k);

}  // namespace xla::gpu::kernel

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_KERNEL_H_
