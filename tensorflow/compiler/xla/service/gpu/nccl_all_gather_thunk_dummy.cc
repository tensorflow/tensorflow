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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace gpu {

NcclAllGatherThunk::NcclAllGatherThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op, int64 replica_count,
    std::vector<NcclAllGatherThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGather, thunk_info), config_{} {}

/* static */ bool NcclAllGatherThunk::CanImplement(const HloInstruction* hlo) {
  return false;
}

/* static */ bool NcclAllGatherThunk::CanImplement(
    mlir::lmhlo::AllGatherOp op) {
  return false;
}

Status NcclAllGatherThunk::RunNcclCollective(const ExecuteParams&, ncclComm_t) {
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
}

const NcclCollectiveConfig& NcclAllGatherThunk::config() const {
  // This function will never be called.
  const NcclCollectiveConfig* config = nullptr;
  return *config;
}

}  // namespace gpu
}  // namespace xla
