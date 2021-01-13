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

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_runner.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#endif

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#endif

namespace xla {
namespace gpu {

std::unique_ptr<Thunk> ThunkEmitter::BuildFftThunk(const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  return absl::make_unique<FftThunk>(
      context_->GetThunkInfo(inst), inst->fft_type(), inst->fft_length(),
      /*input_buffer=*/GetAllocationSlice(*operand),
      /*output_buffer=*/GetAllocationSlice(*inst),
      /*input_shape=*/operand->shape(),
      /*output_shape=*/inst->shape());
}

std::unique_ptr<Thunk> ThunkEmitter::BuildTriangularSolveThunk(
    const HloInstruction* inst) {
  const HloInstruction* a = inst->operand(0);
  const HloInstruction* b = inst->operand(1);
  int64 m = b->shape().dimensions(b->shape().rank() - 2);
  int64 n = b->shape().dimensions(b->shape().rank() - 1);
  int64 batch_size = std::accumulate(
      b->shape().dimensions().begin(), b->shape().dimensions().end() - 2,
      int64{1}, [](int64 a, int64 b) { return a * b; });
  int64 elem_size =
      ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type());
  int64 a_batch_stride = inst->triangular_solve_options().left_side()
                             ? m * m * elem_size
                             : n * n * elem_size;
  int64 b_batch_stride = m * n * elem_size;
  return absl::make_unique<TriangularSolveThunk>(
      context_->GetThunkInfo(inst), inst->triangular_solve_options(),
      /*a_input_buffer=*/GetAllocationSlice(*a),
      /*b_input_buffer=*/GetAllocationSlice(*inst),
      inst->shape().element_type(), batch_size, m, n, a_batch_stride,
      b_batch_stride);
}

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

std::unique_ptr<Thunk> ThunkEmitter::BuildOutfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kOutfeed, inst->opcode());

  const HloInstruction* source = inst->operand(0);
  std::vector<ShapeUtil::IndexedShape> leaf_shapes =
      ShapeUtil::GetLeafShapes(source->shape());

  std::vector<ShapedSlice> source_slices;
  source_slices.reserve(leaf_shapes.size());

  for (ShapeUtil::IndexedShape& indexed_shape : leaf_shapes) {
    BufferAllocation::Slice slice =
        GetAllocationSlice(*source, indexed_shape.index);
    const Shape& shape =
        ShapeUtil::GetSubshape(source->shape(), indexed_shape.index);
    source_slices.push_back(ShapedSlice{slice, shape});
  }

  OutfeedConfig config = GetOutfeedConfig(inst);
  return absl::make_unique<OutfeedThunk>(context_->GetThunkInfo(inst),
                                         std::move(config),
                                         std::move(source_slices));
}

Status ThunkEmitter::HandleCustomCall(HloInstruction* custom_call) {
  // A CustomCall on the GPU backend can either be a custom-call to a
  // user-supplied kernel, or a call into a library like cudnn.

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  if (void* call_target = CustomCallTargetRegistry::Global()->Lookup(
          custom_call->custom_call_target(), std::string(platform_name()))) {
    auto get_slices_for_instr = [&](const HloInstruction* instr) {
      ShapeTree<BufferAllocation::Slice> slices(instr->shape());
      slices.ForEachMutableElement(
          [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
            StatusOr<BufferAllocation::Slice> s =
                MaybeGetAllocationSlice(*instr, index);
            if (s.ok()) {
              *slice = s.ValueOrDie();
            }
          });
      return slices;
    };
    std::vector<ShapeTree<BufferAllocation::Slice>> operand_slices;
    for (int64 i = 0; i < custom_call->operand_count(); i++) {
      const auto* operand = custom_call->operand(i);
      operand_slices.push_back(get_slices_for_instr(operand));
      const auto& s1 = operand_slices.back().shape();
      const auto& s2 = operand->shape();
      CHECK(ShapeUtil::Equal(s1, s2)) << absl::StreamFormat(
          "Shape mismatch between operand shape and "
          "slice shape for operand %d: %s vs %s",
          i, s1.ToString(), s2.ToString());
    }
    ShapeTree<BufferAllocation::Slice> result_slices =
        get_slices_for_instr(custom_call);
    CHECK(ShapeUtil::Equal(custom_call->shape(), result_slices.shape()))
        << absl::StreamFormat(
               "Shape mismatch between instr->shape() and "
               "result_slices.shape(): "
               "%s vs %s.",
               custom_call->shape().ToString(),
               result_slices.shape().ToString());

    AddThunkToThunkSequence(absl::make_unique<CustomCallThunk>(
        context_->GetThunkInfo(custom_call), call_target,
        std::move(operand_slices), std::move(result_slices),
        Cast<HloCustomCallInstruction>(custom_call)->opaque()));
    return Status::OK();
  }
#endif

  return Unimplemented("No registered implementation for custom call to \"%s\"",
                       custom_call->custom_call_target());
}

Status ThunkEmitter::HandleFft(HloInstruction* fft) {
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(fft->operand(0)->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  AddThunkToThunkSequence(BuildFftThunk(fft));
  return Status::OK();
}

Status ThunkEmitter::HandleTriangularSolve(HloInstruction* hlo) {
  auto has_fortran_layout = [](const Layout& layout) {
    int n = layout.minor_to_major_size();
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(hlo->operand(0)->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(hlo->operand(1)->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(hlo->shape().layout()));

  std::vector<std::unique_ptr<Thunk>> thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  auto operand_buffer = GetAllocationSlice(*hlo->operand(1));
  auto destination_buffer = GetAllocationSlice(*hlo);
  if (operand_buffer != destination_buffer) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        context_->GetThunkInfo(hlo),
        /*source_address=*/operand_buffer,
        /*destination_buffer=*/destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(hlo->operand(1)->shape())));
  }

  thunks.push_back(BuildTriangularSolveThunk(hlo));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        context_->GetThunkInfo(hlo), std::move(thunks)));
  }
  return Status::OK();
}

Status ThunkEmitter::HandleOutfeed(HloInstruction* outfeed) {
  AddThunkToThunkSequence(BuildOutfeedThunk(outfeed));
  return Status::OK();
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
