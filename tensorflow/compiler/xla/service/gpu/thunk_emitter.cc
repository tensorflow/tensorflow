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
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#endif

namespace xla {
namespace gpu {

namespace {
void CheckBatchNormInputOutputPrimitivetypeAreValid(const HloInstruction* hlo) {
  // All input and output statistics variables must be F32. Also, the last
  // operand for CudnnBatchNormForwardInference, CudnnBatchNormForwardTraining,
  // and CudnnBatchNormBackward is the feature_index which must be S64.
  // The allowed types for non-statistics variables are as follows:
  // CudnnBatchNormForwardInference:
  //            operand[0]: {half, float}
  //                out[0]: {half, float}
  // CudnnBatchNormForwardTraining:
  //            operand[0]: {half, float}
  //                out[0]: {half, float}
  // CudnnBatchNormBackward:
  //            operand[0]: {half, float}
  //            operand[4]: {half, float}
  //                out[0]: {half, float}
  // Note non-statistics inputs and outputs mentioned above should be of the
  // same type.

  // Check Inputs.
  int64 num_operands = hlo->operand_count();
  PrimitiveType operand_primitive_type =
      hlo->operand(0)->shape().element_type();
  CHECK(operand_primitive_type == F16 || operand_primitive_type == F32)
      << "Not yet implemented";

  for (int i = 1; i < num_operands - 2; i++) {
    if (hlo->custom_call_target() == kCudnnBatchNormBackwardCallTarget &&
        i == 4) {
      // The first operand to batchnorm grad is the input and the 4th operand is
      // the grad_output, both of which can be Eigen::half.
      CHECK_EQ(hlo->operand(i)->shape().element_type(), operand_primitive_type)
          << "Invalid datatype";
      continue;
    }
    CHECK_EQ(hlo->operand(i)->shape().element_type(), F32)
        << "Not yet implemented";
  }

  // The last operand is the feature index which must be int64.
  CHECK_EQ(hlo->operand(num_operands - 1)->shape().element_type(), S64)
      << "Not yet implemented";

  // Check Outputs.
  if (hlo->shape().IsTuple()) {
    CHECK_EQ(hlo->shape().tuple_shapes(0).element_type(),
             operand_primitive_type)
        << "Invalid datatype";

    for (int j = 1; j < hlo->shape().tuple_shapes_size(); j++) {
      CHECK_EQ(hlo->shape().tuple_shapes(j).element_type(), F32)
          << "Not yet implemented";
    }
  } else {
    CHECK_EQ(hlo->shape().element_type(), operand_primitive_type)
        << "Invalid datatype";
  }
}
}  // namespace
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

std::unique_ptr<Thunk> ThunkEmitter::BuildInfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kInfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->shape());
  slices.ForEachMutableElement(
      [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
        *slice = GetAllocationSlice(*inst, index);
      });
  return absl::make_unique<InfeedThunk>(context_->GetThunkInfo(inst), slices);
}

std::unique_ptr<Thunk> ThunkEmitter::BuildOutfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kOutfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->operand(0)->shape());
  slices.ForEachMutableElement([&](const ShapeIndex& index,
                                   BufferAllocation::Slice* slice) {
    auto status_or_slice = MaybeGetAllocationSlice(*inst->operand(0), index);
    if (status_or_slice.ok()) {
      *slice = status_or_slice.ValueOrDie();
    }
  });
  OutfeedConfig config = GetOutfeedConfig(inst);
  return absl::make_unique<OutfeedThunk>(context_->GetThunkInfo(inst),
                                         std::move(config), std::move(slices));
}

Status ThunkEmitter::HandleCustomCall(HloInstruction* custom_call) {
  // A CustomCall on the GPU backend can either be a custom-call to a
  // user-supplied kernel, or a call into a library like cudnn.

  // Lower custom-calls to cudnn batchnorm ops to specialized thunks.  It's part
  // of the contract of these cudnn batchnorm calls that the epsilon and
  // feature_index operands be constants.
  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardInferenceCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(5);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(6);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    CHECK_EQ(custom_call->shape().tuple_shapes_size(), 3);
    CHECK(LayoutUtil::LayoutsInShapesEqual(custom_call->shape().tuple_shapes(0),
                                           custom_call->operand(0)->shape()));
    CheckBatchNormInputOutputPrimitivetypeAreValid(custom_call);
    CudnnBatchNormConfig config = GetCudnnBatchNormConfig(
        custom_call, epsilon_value, feature_index_value);
    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardInferenceThunk>(
            context_->GetThunkInfo(custom_call), std::move(config),
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
            /*offset=*/GetAllocationSlice(*custom_call->operand(2)),
            /*mean=*/GetAllocationSlice(*custom_call->operand(3)),
            /*variance=*/GetAllocationSlice(*custom_call->operand(4)),
            /*output=*/GetAllocationSlice(*custom_call)));
    return Status::OK();
  }

  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardTrainingCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(3);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(4);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    // BatchNormTraining returns a tuple of three elements: data, calculated
    // mean, and calculated 1/sqrt(variance + epsilon).
    auto output_data = GetAllocationSlice(*custom_call, {0});
    auto output_mean = GetAllocationSlice(*custom_call, {1});
    auto output_inv_stddev = GetAllocationSlice(*custom_call, {2});
    CudnnBatchNormConfig config = GetCudnnBatchNormConfig(
        custom_call, epsilon_value, feature_index_value);
    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardTrainingThunk>(
            context_->GetThunkInfo(custom_call), std::move(config),
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
            /*offset=*/GetAllocationSlice(*custom_call->operand(2)),
            /*output_data=*/output_data,
            /*output_mean=*/output_mean,
            /*output_inv_stddev=*/output_inv_stddev,
            /*output_tuple=*/GetAllocationSlice(*custom_call)));
    return Status::OK();
  }

  if (custom_call->custom_call_target() == kCudnnBatchNormBackwardCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(5);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(6);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    // BatchNormGrad returns a tuple of three elements: grad_data, grad_scale,
    // grad_offset.
    auto output_grad_data = GetAllocationSlice(*custom_call, {0});
    auto output_grad_scale = GetAllocationSlice(*custom_call, {1});
    auto output_grad_offset = GetAllocationSlice(*custom_call, {2});
    CHECK_EQ(custom_call->shape().tuple_shapes_size(), 3);
    CHECK(LayoutUtil::LayoutsInShapesEqual(custom_call->shape().tuple_shapes(0),
                                           custom_call->operand(0)->shape()));
    CHECK(LayoutUtil::LayoutsInShapesEqual(custom_call->shape().tuple_shapes(0),
                                           custom_call->operand(4)->shape()));
    CheckBatchNormInputOutputPrimitivetypeAreValid(custom_call);

    CudnnBatchNormConfig config = GetCudnnBatchNormConfig(
        custom_call, epsilon_value, feature_index_value);
    AddThunkToThunkSequence(absl::make_unique<CudnnBatchNormBackwardThunk>(
        context_->GetThunkInfo(custom_call), std::move(config),
        /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
        /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
        /*mean=*/GetAllocationSlice(*custom_call->operand(2)),
        /*inv_stddev=*/GetAllocationSlice(*custom_call->operand(3)),
        /*grad_output=*/GetAllocationSlice(*custom_call->operand(4)),
        /*output_grad_data=*/output_grad_data,
        /*output_grad_scale=*/output_grad_scale,
        /*output_grad_offset=*/output_grad_offset,
        /*output_tuple=*/GetAllocationSlice(*custom_call)));
    return Status::OK();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    std::vector<BufferAllocation::Slice> operand_slices;
    operand_slices.reserve(custom_call->operand_count());
    for (const auto* operand : custom_call->operands()) {
      operand_slices.push_back(GetAllocationSlice(*operand));
    }
    auto tuple_result_slice = GetAllocationSlice(*custom_call);
    auto conv_result_slice = GetAllocationSlice(*custom_call, {0});
    auto scratch_slice = GetAllocationSlice(*custom_call, {1});

    TF_ASSIGN_OR_RETURN(
        GpuConvConfig config,
        GetGpuConvConfig(Cast<HloCustomCallInstruction>(custom_call)));
    AddThunkToThunkSequence(absl::make_unique<ConvolutionThunk>(
        context_->GetThunkInfo(custom_call), std::move(config),
        std::move(operand_slices), conv_result_slice, scratch_slice,
        tuple_result_slice));
    return Status::OK();
  }

  if (IsCublasGemm(*custom_call)) {
    AddThunkToThunkSequence(BuildGemmThunk(custom_call));
    return Status::OK();
  }

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
  if (custom_call->custom_call_target() == kCusolverCholeskyCallTarget) {
    TF_ASSIGN_OR_RETURN(CholeskyOptions options,
                        custom_call->backend_config<CholeskyOptions>());

    const Shape& shape = custom_call->operand(0)->shape();
    int ndim = shape.dimensions_size();
    CHECK_GE(ndim, 2);
    int64 n = shape.dimensions(ndim - 1);

    const auto& dims = shape.dimensions();
    int64 batch_size = std::accumulate(dims.begin(), dims.end() - 2, int64{1},
                                       [](int64 a, int64 b) { return a * b; });

    auto operand_buffer = GetAllocationSlice(*custom_call->operand(0));

    auto a_buffer = GetAllocationSlice(*custom_call, {0});
    auto workspace_buffer = GetAllocationSlice(*custom_call, {1});
    auto info_buffer = GetAllocationSlice(*custom_call, {2});

    std::vector<std::unique_ptr<Thunk>> thunks;

    if (operand_buffer != a_buffer) {
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          context_->GetThunkInfo(custom_call),
          /*source_address=*/operand_buffer,
          /*destination_buffer=*/a_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
    }

    thunks.push_back(absl::make_unique<CholeskyThunk>(
        context_->GetThunkInfo(custom_call), options, a_buffer,
        workspace_buffer, info_buffer,
        custom_call->operand(0)->shape().element_type(), batch_size, n));

    // Elide the sequential thunk if there's no copy.
    if (thunks.size() == 1) {
      AddThunkToThunkSequence(std::move(thunks[0]));
    } else {
      AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
          context_->GetThunkInfo(custom_call), std::move(thunks)));
    }

    return Status::OK();
  }

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

Status ThunkEmitter::HandleInfeed(HloInstruction* infeed) {
  AddThunkToThunkSequence(BuildInfeedThunk(infeed));
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
