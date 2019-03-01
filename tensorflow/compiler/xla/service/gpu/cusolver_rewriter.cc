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

#include "tensorflow/compiler/xla/service/gpu/cusolver_rewriter.h"

#include <cstdlib>
#include <numeric>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/scratch_allocator.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"

namespace xla {
namespace gpu {

namespace {

void SetFortranLayout(Shape* shape) {
  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}

StatusOr<HloInstruction*> CreateCholesky(CusolverContext* context,
                                         ScratchAllocator* allocator,
                                         HloInstruction* operand,
                                         const CholeskyOptions& options,
                                         const OpMetadata& metadata) {
  HloComputation* computation = operand->parent();

  Shape a_shape = operand->shape();
  int ndim = a_shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64 n = a_shape.dimensions(ndim - 1);

  int64 batch_size = std::accumulate(a_shape.dimensions().begin(),
                                     a_shape.dimensions().end() - 2, int64{1},
                                     [](int64 a, int64 b) { return a * b; });

  // Find the workspace size.
  se::blas::UpperLower uplo = options.lower() ? se::blas::UpperLower::kLower
                                              : se::blas::UpperLower::kUpper;
  int64 workspace_size;  // Number of elements of size a_shape.element_type()
  switch (a_shape.element_type()) {
    case F32: {
      TF_ASSIGN_OR_RETURN(auto a,
                          allocator->Allocate<float>(context->stream(), n * n));
      TF_ASSIGN_OR_RETURN(workspace_size,
                          context->PotrfBufferSize(uplo, n, a, n));
      break;
    }
    case F64: {
      TF_ASSIGN_OR_RETURN(
          auto a, allocator->Allocate<double>(context->stream(), n * n));
      TF_ASSIGN_OR_RETURN(workspace_size,
                          context->PotrfBufferSize(uplo, n, a, n));
      break;
    }
    case C64: {
      TF_ASSIGN_OR_RETURN(auto a, allocator->Allocate<std::complex<float>>(
                                      context->stream(), n * n));
      TF_ASSIGN_OR_RETURN(workspace_size,
                          context->PotrfBufferSize(uplo, n, a, n));
      break;
    }
    case C128: {
      TF_ASSIGN_OR_RETURN(auto a, allocator->Allocate<std::complex<double>>(
                                      context->stream(), n * n));
      TF_ASSIGN_OR_RETURN(workspace_size,
                          context->PotrfBufferSize(uplo, n, a, n));
      break;
    }
    default:
      return InvalidArgument("Invalid type for cholesky decomposition: %s",
                             a_shape.ToString());
  }

  // TODO(phawkins): Ideally we would relax this constraint. What we actually
  // want is that:
  // a) the batch dimensions are major, in no particular order.
  // b) the two minor dimensions are in fortran (column-major) order,

  SetFortranLayout(&a_shape);

  // This call returns a tuple of (cholesky_result, workspace, info) where:
  // * cholesky_result is the result of the Cholesky decomposition,
  // * workspace is temporary scratch memory used by cuSolver.
  // * info contains the Potrf success/failure status.
  // Currently we have no meaningful way to report an error, so we simply
  // discard the success/failure information. Obviously this is suboptimal.
  Shape call_shape = ShapeUtil::MakeTupleShape(
      {a_shape,
       ShapeUtil::MakeShape(operand->shape().element_type(), {workspace_size}),
       ShapeUtil::MakeShape(S32, {batch_size})});

  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, {operand}, kCusolverCholeskyCallTarget, {a_shape}));
  custom_call->set_metadata(metadata);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(options));
  return custom_call;
}

}  // namespace

// Tries to rewrite a single convolution into a call to cudnn.
StatusOr<bool> RunOnInstruction(CusolverContext* context,
                                ScratchAllocator* allocator,
                                HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCholesky) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * custom_call,
      CreateCholesky(context, allocator, instruction->mutable_operand(0),
                     instruction->cholesky_options(), instruction->metadata()));

  VLOG(1) << "Replacing " << instruction->ToString() << " with "
          << custom_call->ToString();

  // The CustomCall returns a tuple (conv_result, scratch_memory).  Extract out
  // the conv result and replace `conv` with it.
  TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
      instruction, HloInstruction::CreateGetTupleElement(instruction->shape(),
                                                         custom_call, 0)));
  return true;
}

// Rewrites the convolutions in the given computation into calls to cudnn.
// Returns true if it made any changes.
StatusOr<bool> CusolverRewriter::RunOnComputation(HloComputation* computation) {
  std::vector<HloInstruction*> cusolver_calls;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kCholesky) {
      cusolver_calls.push_back(hlo);
    }
  }

  if (cusolver_calls.empty()) {
    return false;
  }

  // Create a stream for us to do our work on. We don't really need to do any
  // work, just allocate memory, but that's the cuSolver API.
  se::Stream stream{stream_exec_};
  stream.Init();
  const auto device_ordinal = stream_exec_->device_ordinal();

  // allocator either points to this->allocator_ or, if that's null, to a
  // StreamExecutorMemoryAllocator for stream_exec_.
  DeviceMemoryAllocator* allocator;
  absl::optional<StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_->platform(),
                         absl::Span<se::StreamExecutor* const>({stream_exec_}));
    allocator = &*se_allocator;
  }
  ScratchAllocator scratch_allocator(device_ordinal, allocator);

  TF_ASSIGN_OR_RETURN(CusolverContext context,
                      CusolverContext::Create(&stream));

  bool changed = false;
  for (HloInstruction* instruction : cusolver_calls) {
    TF_ASSIGN_OR_RETURN(
        bool result,
        RunOnInstruction(&context, &scratch_allocator, instruction));
    changed |= result;
  }
  return changed;
}

CusolverRewriter::CusolverRewriter(se::StreamExecutor* stream_exec,
                                   DeviceMemoryAllocator* allocator)
    : stream_exec_(stream_exec), allocator_(allocator) {}

StatusOr<bool> CusolverRewriter::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
