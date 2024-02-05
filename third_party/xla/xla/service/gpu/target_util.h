/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TARGET_UTIL_H_
#define XLA_SERVICE_GPU_TARGET_UTIL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Enumeration to get target specific intrinsics.
enum class TargetIntrinsicID {
  kThreadIdx = 0,
  kThreadIdy,
  kThreadIdz,
  kBlockIdx,
  kBlockIdy,
  kBlockIdz,
  kBarrierId,
  kBlockDimx,
  kBlockDimy,
  kBlockDimz,
  kGroupBarrierId,
};

// Enumeration to get target specific device math function.
enum class TargetDeviceFunctionID {
  kAtan2 = 0,
  kCbrt,
  kCos,
  kExp,
  kExpm1,
  kFmod,
  kHypot,
  kLog,
  kLog1p,
  kPow,
  kRsqrt,
  kSin,
  kSqrt,
  kTan,
  kTanh,
  kErf,
};

// HLO opcode -> TargetDeviceFunctionID mapping.
absl::StatusOr<TargetDeviceFunctionID> GetTargetDeviceFunctionID(HloOpcode);

// Emits IR to call a device function named "callee_name" on the given
// operand. Returns the IR value that represents the return value.
llvm::CallInst* EmitDeviceFunctionCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_type, PrimitiveType output_type,
    const llvm::AttrBuilder& attributes, llvm::IRBuilder<>* b,
    absl::string_view name = "");

// Emits a call to the specified target intrinsic with the given operands.
// Overloaded intrinsics (for example, "minnum") must include a type
// in overloaded_types  for each overloaded type. Typically, overloaded
// intrinsics have only a single overloaded type.
llvm::CallInst* EmitCallToTargetIntrinsic(
    TargetIntrinsicID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b);

// Annotate the kernel as GPU kernel according to the GPU target.
void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilder<>* b);

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     llvm::Triple target_triple);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TARGET_UTIL_H_
