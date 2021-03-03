/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace gpu {

// Thunk that implements the collective-permute HLO.
class CollectivePermuteThunk : public Thunk {
 public:
  CollectivePermuteThunk(
      ThunkInfo thunk_info,
      std::vector<std::pair<int64, int64>> source_target_pairs,
      const BufferAllocation::Slice& src, const BufferAllocation::Slice& dest);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<std::pair<int64, int64>> source_target_pairs_;
  const BufferAllocation::Slice src_;
  const BufferAllocation::Slice dest_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COLLECTIVE_PERMUTE_THUNK_H_
