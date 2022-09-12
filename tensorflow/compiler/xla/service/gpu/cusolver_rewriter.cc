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
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/tsl/platform/logging.h"

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

StatusOr<HloInstruction*> CreateCholesky(GpuSolverContext* context,
                                         HloInstruction* operand,
                                         const CholeskyOptions& options,
                                         const OpMetadata& metadata) {
  HloComputation* computation = operand->parent();

  Shape a_shape = operand->shape();
  int ndim = a_shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64_t n = a_shape.dimensions(ndim - 1);

  std::vector<int64_t> batch_dims(a_shape.dimensions().begin(),
                                  a_shape.dimensions().end() - 2);
  std::vector<int64_t> batch_dim_ids(batch_dims.size());
  absl::c_iota(batch_dim_ids, 0);
  int64_t batch_size = absl::c_accumulate(batch_dims, 1, std::multiplies<>{});

  // Find the workspace size.
  se::blas::UpperLower uplo = options.lower() ? se::blas::UpperLower::kLower
                                              : se::blas::UpperLower::kUpper;
  int64_t workspace_size;  // Number of elements of size a_shape.element_type()
  TF_ASSIGN_OR_RETURN(
      workspace_size,
      context->PotrfBufferSize(a_shape.element_type(), uplo, n, n, batch_size));

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
  Shape info_shape = ShapeUtil::MakeShape(S32, batch_dims);
  Shape call_shape = ShapeUtil::MakeTupleShape(
      {a_shape,
       ShapeUtil::MakeShape(operand->shape().element_type(), {workspace_size}),
       info_shape});

  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, {operand}, kCusolverCholeskyCallTarget, {a_shape}));
  custom_call->set_metadata(metadata);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(options));
  HloInstruction* out = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(a_shape, custom_call, 0));
  HloInstruction* info = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(info_shape, custom_call, 2));

  // If info was non-zero, indicating that the Cholesky decomposition failed,
  // returns an array full of NaNs for the corresponding batch element.
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* zeros =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          info_shape, zero, /*broadcast_dimensions=*/{}));
  HloInstruction* ok = computation->AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, batch_dims),
                                    info, zeros, ComparisonDirection::kEq));
  ok = computation->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, a_shape.dimensions()), ok,
      /*broadcast_dimensions=*/batch_dim_ids));

  TF_ASSIGN_OR_RETURN(Literal nan_literal,
                      LiteralUtil::NanValue(a_shape.element_type()));
  HloInstruction* nan = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(nan_literal)));
  HloInstruction* nans =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          a_shape, nan, /*broadcast_dimensions=*/{}));

  HloInstruction* select =
      computation->AddInstruction(HloInstruction::CreateTernary(
          a_shape, HloOpcode::kSelect, ok, out, nans));
  return select;
}

// Tries to rewrite a single convolution into a call to cudnn.
StatusOr<bool> RunOnInstruction(GpuSolverContext* context,
                                HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCholesky) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * custom_call,
      CreateCholesky(context, instruction->mutable_operand(0),
                     instruction->cholesky_options(), instruction->metadata()));

  VLOG(1) << "Replacing " << instruction->ToString() << " with "
          << custom_call->ToString();

  TF_RETURN_IF_ERROR(
      instruction->parent()->ReplaceInstruction(instruction, custom_call));
  return true;
}

}  // namespace

// Rewrites the convolutions in the given computation into calls to cudnn.
// Returns true if it made any changes.
StatusOr<bool> GpusolverRewriter::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> cusolver_calls;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kCholesky) {
      cusolver_calls.push_back(hlo);
    }
  }

  if (cusolver_calls.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(GpuSolverContext context,
                      GpuSolverContext::Create(/*stream=*/nullptr));

  bool changed = false;
  for (HloInstruction* instruction : cusolver_calls) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(&context, instruction));
    changed |= result;
  }
  return changed;
}

GpusolverRewriter::GpusolverRewriter() = default;

StatusOr<bool> GpusolverRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
