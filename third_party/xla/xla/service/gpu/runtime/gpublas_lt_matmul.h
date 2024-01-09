/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_H_
#define XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_H_

#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call_registry.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/matmul_utils.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

// Add cuBLASLt attributes encoding
void PopulateCublasLtMatmulAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

#if GOOGLE_CUDA || TF_HIPBLASLT

// Registers XLA Gpu runtime cuBLASLt custom calls.
void RegisterMatmulCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Keep cublas_lt::MatmulPlan's for all matmul instances in the executable.
class MatmulPlans
    : public runtime::StateVector<se::gpu::BlasLt::MatmulPlanPtr> {};
#endif  // GOOGLE_CUDA || TF_HIPBLASLT

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_H_
