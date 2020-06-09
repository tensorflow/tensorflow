/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {

llvm::TargetTransformInfo* LLVMTargetMachineFeatures::GetTargetTransformInfoFor(
    const llvm::Function& function) const {
  auto it = target_transform_info_cache_.find(&function);
  if (it == target_transform_info_cache_.end()) {
    auto emplace_result = target_transform_info_cache_.emplace(
        &function, target_machine_->getTargetTransformInfo(function));
    CHECK(emplace_result.second);
    it = emplace_result.first;
  }

  return &it->second;
}

int64 LLVMTargetMachineFeatures::minimum_alignment_for_allocation(
    int64 size_bytes) const {
  // Assume that all pointers are aligned to at least
  // xla::cpu_function_runtime::kMinAlign.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  // Allow small buffers to be underaligned, there is no vectorization benefit
  // anyways.
  return std::min<int64>(llvm::PowerOf2Ceil(size_bytes),
                         cpu_function_runtime::kMinAlign);
}

}  // namespace cpu
}  // namespace xla
