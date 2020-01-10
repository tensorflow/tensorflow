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
  // GLibc malloc returns a pointer with alignment 8 on 32-bit platforms and 16
  // on 64-bit platforms.  TCMalloc returns a pointer with alignment 8 for
  // allocations smaller than kMallocAlignmentThreshold bytes and at least
  // alignment 16 for allocations greater than or equal to
  // kMallocAlignmentThreshold bytes.  N.B. We could improve on this lower bound
  // by explicitly allocating the memory with posix_memalign.  This is
  // complicated by our desire to allow parameter buffers created by clients to
  // be consumed directly by the JIT.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  const int64 kMallocAlignmentThreshold = 512;

  int pointer_size = target_machine_->getPointerSize(0);
  int buffer_alignment =
      size_bytes >= kMallocAlignmentThreshold ? 2 * pointer_size : pointer_size;
  DCHECK_GT(buffer_alignment, 0);

  return buffer_alignment;
}

}  // namespace cpu
}  // namespace xla
