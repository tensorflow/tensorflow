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

#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace gpu {

std::unique_ptr<Thunk> ThunkEmitter::BuildGemmThunk(
    const HloInstruction* inst) {
  GpuGemmConfig config = GetGpuGemmConfig(inst);
  const HloInstruction* lhs = inst->operand(0);
  const HloInstruction* rhs = inst->operand(1);

  // The bias is passed inside the output buffer. If those buffers are shared
  // we can just use it, otherwise copy the bias values into the output buffer
  // first.
  if (config.backend_config.beta() != 0.0) {
    const HloInstruction* bias = inst->operand(2);
    CHECK_EQ(bias->shape(), inst->shape());
    if (GetAllocationSlice(*bias) != GetAllocationSlice(*inst)) {
      std::vector<std::unique_ptr<Thunk>> thunks;
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo(),
          /*source_buffer=*/GetAllocationSlice(*bias),
          /*destination_buffer=*/GetAllocationSlice(*inst),
          /*mem_size=*/ShapeUtil::ByteSizeOf(inst->shape())));
      thunks.push_back(absl::make_unique<GemmThunk>(
          context_->GetThunkInfo(inst), std::move(config),
          GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
          GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
          GetAllocationSlice(*inst),  // The output buffer.
          /*implements_whole_instruction=*/false));
      return absl::make_unique<SequentialThunk>(context_->GetThunkInfo(inst),
                                                std::move(thunks));
    }
  }

  return absl::make_unique<GemmThunk>(
      context_->GetThunkInfo(inst), std::move(config),
      GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
      GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
      GetAllocationSlice(*inst),  // The output buffer.
      /*implements_whole_instruction=*/true);
}

Thunk::ThunkInfo ThunkEmitter::EmissionContext::GetThunkInfo(
    const HloInstruction* hlo) const {
  CHECK(hlo);
  Thunk::ThunkInfo info;
  info.profile_annotation = absl::StrFormat(
      "Thunk:#hlo_op=%s,hlo_module=%s#", hlo->name(), hlo->GetModule()->name());
  return info;
}
}  // namespace gpu
}  // namespace xla
