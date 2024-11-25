/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/target_machine_features.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/primitive_util.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

TargetMachineFeatures::TargetMachineFeatures(
    llvm::TargetMachine* target_machine)
    : target_machine_(target_machine) {}

int32_t TargetMachineFeatures::vectorization_factor_in_bytes() const {
  // Ideally this should be a function of the cache line size (which we can
  // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
  // machine.  Guess a value of 128 bytes for now.
  return 128;
}

int32_t TargetMachineFeatures::vector_register_byte_size(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return tti->getRegisterBitWidth(
             llvm::TargetTransformInfo::RGK_FixedWidthVector) /
         8;
}

int32_t TargetMachineFeatures::vector_register_num_elements(
    const llvm::Function& fn, PrimitiveType type) const {
  return vector_register_byte_size(fn) / (primitive_util::BitWidth(type) / 8);
}

int32_t TargetMachineFeatures::vector_register_count(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return static_cast<int32_t>(
      tti->getNumberOfRegisters(tti->getRegisterClassForType(/*Vector=*/true)));
}

llvm::TargetTransformInfo* TargetMachineFeatures::GetTargetTransformInfoFor(
    const llvm::Function& fn) const {
  auto it = target_transform_info_.find(&fn);
  if (it == target_transform_info_.end()) {
    auto emplace_result = target_transform_info_.emplace(
        &fn, target_machine_->getTargetTransformInfo(fn));
    CHECK(emplace_result.second);
    it = emplace_result.first;
  }

  return &it->second;
}

int64_t TargetMachineFeatures::minimum_alignment_for_allocation(
    int64_t size_bytes) const {
  // Assume that all pointers are aligned to at least
  // xla::cpu_function_runtime::kMinAlign.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  // Allow small buffers to be underaligned, there is no vectorization benefit
  // anyways.
  return std::min<int64_t>(llvm::PowerOf2Ceil(size_bytes), MinAlign());
}

std::string TargetMachineFeatures::get_target_feature_string() const {
  return target_machine_->getTargetFeatureString().str();
}

}  // namespace xla::cpu
