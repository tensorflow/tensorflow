/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_CUBLAS_PADDING_REQUIREMENTS_H_
#define XLA_SERVICE_GPU_CUBLAS_PADDING_REQUIREMENTS_H_

#include <array>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

struct CublasPaddingRequirement {
  int min_compute_capability;
  PrimitiveType data_type;
  int multiple_of;
};

struct HipblasPaddingRequirement {
  PrimitiveType data_type;
  int multiple_of;
};

// List of padding requirements per compute capability and data type.
constexpr std::array<CublasPaddingRequirement, 3> CublasPaddingRequirements{
    {{se::CudaComputeCapability::kVolta, S8, 4},
     {se::CudaComputeCapability::kVolta, F16, 8},
     {se::CudaComputeCapability::kAmpere, BF16, 8}}};

constexpr std::array<HipblasPaddingRequirement, 2> HipblasPaddingRequirements{
    {{/*rocm gpu arch,*/ F16, 8}, {/*rocm gpu arch,*/ BF16, 8}}};

// Tell if either of the operands of the dot requires padding.
bool CublasRequiresPadding(const HloDotInstruction& dot,
                           const se::GpuComputeCapability& cc);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUBLAS_PADDING_REQUIREMENTS_H_
