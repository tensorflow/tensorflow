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
#ifndef XLA_SERVICE_GPU_TRITON_SUPPORT_H_
#define XLA_SERVICE_GPU_TRITON_SUPPORT_H_

// This file is the home of the basic Triton support checks which are used by
// multiple other components.

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
using CodegenDecision = FusionDecision;

// Tells if f(a+b) == f(a) + f(b).
bool IsDistributiveOverAddition(const HloInstruction& hlo);

// Allowlist of unary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedUnaryElementwise(PrimitiveType);

// Allowlist of binary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedBinaryElementwise(PrimitiveType);

// Allowlist of ternary elementwise operations supported by Triton GEMM codegen.
std::vector<HloOpcode> TritonSupportedTernaryElementwise(PrimitiveType);

// Data types that are supported by the Triton emitters.
bool IsTritonSupportedDataType(PrimitiveType, const se::GpuComputeCapability&);

// Checks elementwise operation against all supported by Triton GEMM codegen.
bool IsTritonSupportedElementwise(HloOpcode, PrimitiveType);

CodegenDecision CanTritonHandleGEMM(
    const HloDotInstruction& dot, const se::GpuComputeCapability& gpu_version);

// Checks instruction against requirements of triton emitter.
CodegenDecision IsTritonSupportedInstruction(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version);

// Checks dynamic slice against requirements of triton emitter.
//
// This is exposed separately from IsTritonSupportedInstruction because we can
// use it in the dimension order propagation without adding a dependency on the
// GPU version.
CodegenDecision IsTritonSupportedDynamicSlice(
    const HloDynamicSliceInstruction& instr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRITON_SUPPORT_H_
