/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/cudnn_fused_mha_transpose_fusion.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

bool IsFMHACustomCall(const HloInstruction* instr) {
  return IsCustomCallTofMHA(*instr);
}

bool IsFwdFMHACustomCall(const HloInstruction* instr) {
  return IsFwdCustomCallTofMHA(*instr);
}

bool IsBwdFMHACustomCall(const HloInstruction* instr) {
  return IsBwdCustomCallTofMHA(*instr);
}

absl::StatusOr<bool> FuseArgPrologueTransposeWithcuDNNFMHA(
    HloInstruction* fmha, int64_t operand_index, bool is_lhs,
    bool should_contracting_be_fastest) {
  HloInstruction* transpose_arg = fmha->mutable_operand(operand_index);
  HloInstruction* transpose_arg_operand = transpose_arg->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      fmha->backend_config<GpuBackendConfig>());
  CudnnfMHABackendConfig config = gpu_config.cudnn_fmha_backend_config();
  CudnnfMHABackendConfig& new_fmha_config =
      *gpu_config.mutable_cudnn_fmha_backend_config();

  std::vector<int64_t> inverse_perm =
      InversePermutation(transpose_arg->dimensions());
  DotDimensionNumbers new_bmm_dot_dims;
  if (IsFwdCustomCallTofMHA(*fmha)) {
    if (operand_index == 0 || operand_index == 1) {
      new_bmm_dot_dims = config.bmm1_dot_dimension_numbers();
    } else {
      new_bmm_dot_dims = config.bmm2_dot_dimension_numbers();
    }
  } else {
    switch (operand_index) {
      case 0:
        // Q
        new_bmm_dot_dims = config.bmm1_grad_gemm1_dot_dimension_numbers();
        break;
      case 1:
        // K
        new_bmm_dot_dims = config.bmm1_grad_gemm2_dot_dimension_numbers();
        break;
      case 2:
        // V
        new_bmm_dot_dims = config.bmm2_grad_gemm2_dot_dimension_numbers();
        break;
      case 3:
        // Forward activation
        new_bmm_dot_dims = config.bmm2_grad_gemm1_dot_dimension_numbers();
        break;
      case 4:
        // Output gradient
        new_bmm_dot_dims = config.bmm2_grad_gemm2_dot_dimension_numbers();
        break;
      default:
        return Internal("Invalid operand index.");
    }
  }
  absl::Span<const int64_t> checked_dims;
  std::vector<int64_t> checked_dims_vec;

  // `should_contracting_be_fastest` means if contracting dim is the head
  // dim. cuDNN requires head dim to be the fastest dim. fwd bmm1 and bwd
  // bmm2grad1 should set this value to true.
  if (should_contracting_be_fastest) {
    checked_dims = is_lhs ? new_bmm_dot_dims.lhs_contracting_dimensions()
                          : new_bmm_dot_dims.rhs_contracting_dimensions();
  } else {
    absl::Span<const int64_t> batch_dims =
        is_lhs ? new_bmm_dot_dims.lhs_batch_dimensions()
               : new_bmm_dot_dims.rhs_batch_dimensions();
    absl::Span<const int64_t> contracting_dims =
        is_lhs ? new_bmm_dot_dims.lhs_contracting_dimensions()
               : new_bmm_dot_dims.rhs_contracting_dimensions();

    TF_ASSIGN_OR_RETURN(checked_dims_vec,
                        GetNonContractingDims(transpose_arg->shape(),
                                              batch_dims, contracting_dims));
    checked_dims = checked_dims_vec;
  }

  int64_t checked_dims_bmm_size = checked_dims.size();
  std::vector<int64_t> new_bmm_checked_dims(checked_dims_bmm_size);
  for (int i = 0; i < checked_dims_bmm_size; i++) {
    auto itr =
        std::find(inverse_perm.begin(), inverse_perm.end(), checked_dims[i]);
    if (itr == inverse_perm.end()) {
      return Internal("Invalid inverse perm");
    }
    new_bmm_checked_dims[i] = std::distance(inverse_perm.begin(), itr);
  }
  // We want to make sure that making the argument to transpose, an input to
  // fmha, doesn't break cuDNN constraint that the head dim of
  // corresponding operand of BMM is the fastest moving dimension.
  // One exception is the forward activation which doesn't have the constraint
  // since it does not have head dim.
  absl::Span<const int64_t> minor_to_major_bmm =
      transpose_arg_operand->shape().layout().minor_to_major();
  if ((minor_to_major_bmm[0] != new_bmm_checked_dims[0]) &&
      !(IsBwdCustomCallTofMHA(*fmha) && operand_index == 3)) {
    return false;
  }
  if (should_contracting_be_fastest) {
    if (is_lhs) {
      new_bmm_dot_dims.clear_lhs_contracting_dimensions();
      *new_bmm_dot_dims.mutable_lhs_contracting_dimensions() = {
          new_bmm_checked_dims.begin(), new_bmm_checked_dims.end()};
    } else {
      new_bmm_dot_dims.clear_rhs_contracting_dimensions();
      *new_bmm_dot_dims.mutable_rhs_contracting_dimensions() = {
          new_bmm_checked_dims.begin(), new_bmm_checked_dims.end()};
    }
  }
  auto& batch_dims = is_lhs ? new_bmm_dot_dims.lhs_batch_dimensions()
                            : new_bmm_dot_dims.rhs_batch_dimensions();
  int64_t batch_dims_bmm_size = batch_dims.size();
  std::vector<int64_t> new_bmm_batch_dims(batch_dims_bmm_size);
  for (int i = 0; i < batch_dims_bmm_size; i++) {
    auto itr =
        std::find(inverse_perm.begin(), inverse_perm.end(), batch_dims[i]);
    if (itr == inverse_perm.end()) {
      return Internal("Invalid inverse perm");
    }
    new_bmm_batch_dims[i] = std::distance(inverse_perm.begin(), itr);
  }

  if (is_lhs) {
    new_bmm_dot_dims.clear_lhs_batch_dimensions();
    *new_bmm_dot_dims.mutable_lhs_batch_dimensions() = {
        new_bmm_batch_dims.begin(), new_bmm_batch_dims.end()};

  } else {
    new_bmm_dot_dims.clear_rhs_batch_dimensions();
    *new_bmm_dot_dims.mutable_rhs_batch_dimensions() = {
        new_bmm_batch_dims.begin(), new_bmm_batch_dims.end()};
  }

  if (!should_contracting_be_fastest) {
    // Given the non-contracting dimensions, we can use the same function,
    // GetNonContractingDims, to find the new contracting dims. Simply pass the
    // non-contracting dimensions as the second argument.
    TF_ASSIGN_OR_RETURN(
        std::vector<int64_t> new_bmm_contracting_dims,
        GetNonContractingDims(transpose_arg_operand->shape(),
                              new_bmm_batch_dims, new_bmm_checked_dims));
    if (is_lhs) {
      new_bmm_dot_dims.clear_lhs_contracting_dimensions();
      *new_bmm_dot_dims.mutable_lhs_contracting_dimensions() = {
          new_bmm_contracting_dims.begin(), new_bmm_contracting_dims.end()};

    } else {
      new_bmm_dot_dims.clear_rhs_contracting_dimensions();
      *new_bmm_dot_dims.mutable_rhs_contracting_dimensions() = {
          new_bmm_contracting_dims.begin(), new_bmm_contracting_dims.end()};
    }
  }
  if (IsFwdCustomCallTofMHA(*fmha)) {
    if (operand_index == 0 || operand_index == 1) {
      // Q or K
      *new_fmha_config.mutable_bmm1_dot_dimension_numbers() = new_bmm_dot_dims;
    } else {
      // V
      *new_fmha_config.mutable_bmm2_dot_dimension_numbers() = new_bmm_dot_dims;
    }
  } else {
    switch (operand_index) {
      case 0:
        // Q
        *new_fmha_config.mutable_bmm1_grad_gemm1_dot_dimension_numbers() =
            new_bmm_dot_dims;
        break;
      case 1:
        // K
        *new_fmha_config.mutable_bmm1_grad_gemm2_dot_dimension_numbers() =
            new_bmm_dot_dims;
        break;
      case 2:
        // V
        *new_fmha_config.mutable_bmm2_grad_gemm2_dot_dimension_numbers() =
            new_bmm_dot_dims;
        break;
      case 3:
        // Forward activation
        *new_fmha_config.mutable_bmm2_grad_gemm1_dot_dimension_numbers() =
            new_bmm_dot_dims;
        break;
      case 4: {
        // Output gradient
        *new_fmha_config.mutable_bmm2_grad_gemm2_dot_dimension_numbers() =
            new_bmm_dot_dims;
        DotDimensionNumbers bmm2_grad_gemm1_dot_dims =
            config.bmm2_grad_gemm1_dot_dimension_numbers();
        absl::Span<const int64_t> bmm2_grad_gemm1_contracting_dims =
            bmm2_grad_gemm1_dot_dims.rhs_contracting_dimensions();
        CHECK_EQ(bmm2_grad_gemm1_contracting_dims.size(), 1);
        absl::Span<const int64_t> transpose_permutation =
            transpose_arg->dimensions();
        auto itr = std::find(transpose_permutation.begin(),
                             transpose_permutation.end(),
                             bmm2_grad_gemm1_contracting_dims[0]);
        if (itr == transpose_permutation.end()) {
          return Internal(
              "bmm2 gradident gemm1 contracting dimension not found.");
        }
        int64_t index = std::distance(transpose_permutation.begin(), itr);
        std::vector<int64_t> new_bmm2_grad_gemm1_rhs_contracting_dims = {index};
        // Find the new batch dimensions, this is done by passing new
        // contracting dimensions and contracting dimension of lhs of
        // bmm2_grad_gemm2(which is the non-contracting dimension of rhs
        // bmm2_grad_gemm1) to GetNonContractingDims.
        TF_ASSIGN_OR_RETURN(
            std::vector<int64_t> new_bmm2_grad_gemm1_rhs_batch_dims,
            GetNonContractingDims(
                transpose_arg_operand->shape(),
                new_bmm2_grad_gemm1_rhs_contracting_dims,
                new_bmm_dot_dims.lhs_contracting_dimensions()));
        bmm2_grad_gemm1_dot_dims.clear_rhs_contracting_dimensions();
        bmm2_grad_gemm1_dot_dims.clear_rhs_batch_dimensions();
        *bmm2_grad_gemm1_dot_dims.mutable_rhs_contracting_dimensions() = {
            new_bmm2_grad_gemm1_rhs_contracting_dims.begin(),
            new_bmm2_grad_gemm1_rhs_contracting_dims.end()};
        *bmm2_grad_gemm1_dot_dims.mutable_rhs_batch_dimensions() = {
            new_bmm2_grad_gemm1_rhs_batch_dims.begin(),
            new_bmm2_grad_gemm1_rhs_batch_dims.end()};
        *new_fmha_config.mutable_bmm2_grad_gemm1_dot_dimension_numbers() =
            bmm2_grad_gemm1_dot_dims;
        break;
      }
      default:
        return Internal("Invalid operand index.");
    }
  }

  TF_RETURN_IF_ERROR(fmha->set_backend_config(gpu_config));

  TF_RETURN_IF_ERROR(fmha->ReplaceOperandWithDifferentShape(
      operand_index, transpose_arg_operand));

  return true;
}

/* Let's say A is transposed to B with perm {3, 0, 2, 1} as shown below:
A[16, 256, 32, 64]
      |
      |
      | Transpose with perm = {3, 0, 2, 1}
      |
      \/
B[64, 16, 32, 256]

The inverse perm to obtain A from B would be {1, 3, 2, 0}. That is
B[64, 16, 32, 256]
      |
      |
      | Transpose' with inv_perm = {1, 3, 2, 0}
      |
      \/
A[16, 256, 32, 64]

Now, let's say B is the lhs of a BatchedMatmul and the lhs_contracting
dim is 3 (i.e dim 256). In order to now make A the lhs to the
batchedMatmul (thus consuming the Transpose from A->B), we need to find
the dimension number in A that corresponds to dimension number 3 in B.
This can be done by finding the index of dim num 3 in inv_perm. That
would be 2. Hence, dim num 3 in B is equivalent to dim num 2 in A. Thus
the new lhs_contracting dim ,if A were to be the new lhs, would be 2.

Similarly, we need to find corresponding batch dimensions as well.
*/
absl::StatusOr<bool> FusePrologueTransposeWithcuDNNFMHA(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *transpose_arg0, *transpose_arg0_operand;
    HloInstruction *transpose_arg1, *transpose_arg1_operand;
    HloInstruction *transpose_arg2, *transpose_arg2_operand;
    HloInstruction *transpose_arg3, *transpose_arg3_operand;
    HloInstruction *transpose_arg4, *transpose_arg4_operand;

    HloInstruction* fmha;

    // Arg0 is common between forward and backward fmha calls, so we match
    // either of these.
    auto pattern_arg0 =
        m::Op(&fmha)
            .WithPredicate(IsFMHACustomCall)
            .WithOperand(0, m::Transpose(&transpose_arg0,
                                         m::Op(&transpose_arg0_operand)));
    if (Match(instr, pattern_arg0)) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Before CudnnFusedMHATransposeFusion Arg 0: \n"
                << comp->parent()->ToString();
      }
      if (IsFwdFMHACustomCall(fmha)) {
        // Q tensor in forward graph is lhs with constraint on contracting dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 0, true /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
      } else {
        // Q tensor in backward graph is rhs with constraint on non-contracting
        // dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 0, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
      }

      if (changed && VLOG_IS_ON(2)) {
        VLOG(2) << "After CudnnFusedMHATransposeFusion Arg 0: \n"
                << comp->parent()->ToString();
      }
    }

    // Arg1 is common between forward and backward fmha calls, so we match
    // either of these.
    auto pattern_arg1 =
        m::Op(&fmha)
            .WithPredicate(IsFMHACustomCall)
            .WithOperand(1, m::Transpose(&transpose_arg1,
                                         m::Op(&transpose_arg1_operand)));
    if (Match(instr, pattern_arg1)) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Before CudnnFusedMHATransposeFusion Arg 1: \n"
                << comp->parent()->ToString();
      }
      if (IsFwdFMHACustomCall(fmha)) {
        // K tensor in forward graph is rhs with constraint on contracting dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 1, false /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
      } else {
        // K tensor in backward graph is rhs with constraint on non-contracting
        // dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 1, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
      }

      if (changed && VLOG_IS_ON(2)) {
        VLOG(2) << "After CudnnFusedMHATransposeFusion Arg 1: \n"
                << comp->parent()->ToString();
      }
    }

    // Arg2 is common between forward and backward fmha calls, so we match
    // either of these.
    auto pattern_arg2 =
        m::Op(&fmha)
            .WithPredicate(IsFMHACustomCall)
            .WithOperand(2, m::Transpose(&transpose_arg2,
                                         m::Op(&transpose_arg2_operand)));
    if (Match(instr, pattern_arg2)) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Before CudnnFusedMHATransposeFusion Arg 2: \n"
                << comp->parent()->ToString();
      }
      if (IsFwdFMHACustomCall(fmha)) {
        // V tensor in forward graph is rhs with constraint on non-contracting
        // dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 2, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
      } else {
        // V tensor in backward graph is rhs with constraint on contracting dim.
        TF_ASSIGN_OR_RETURN(changed,
                            FuseArgPrologueTransposeWithcuDNNFMHA(
                                fmha, 2, false /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
      }

      if (changed && VLOG_IS_ON(2)) {
        VLOG(2) << "After CudnnFusedMHATransposeFusion Arg 2: \n"
                << comp->parent()->ToString();
      }
    }

    // We only care about arg3 of backward
    auto pattern_arg3 =
        m::Op(&fmha)
            .WithPredicate(IsBwdFMHACustomCall)
            .WithOperand(3, m::Transpose(&transpose_arg3,
                                         m::Op(&transpose_arg3_operand)));
    if (Match(instr, pattern_arg3)) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Before CudnnFusedMHATransposeFusion Arg 3: \n"
                << comp->parent()->ToString();
      }
      // Forward activation tensor in backward graph is lhs with constraint on
      // non-contracting dim.
      TF_ASSIGN_OR_RETURN(changed,
                          FuseArgPrologueTransposeWithcuDNNFMHA(
                              fmha, 3, true /*is_lhs*/,
                              false /*should_contracting_be_fastest*/));

      if (changed && VLOG_IS_ON(2)) {
        VLOG(2) << "After CudnnFusedMHATransposeFusion Arg 3: \n"
                << comp->parent()->ToString();
      }
    }

    // We only care about arg4 of backward
    auto pattern_arg4 =
        m::Op(&fmha)
            .WithPredicate(IsBwdFMHACustomCall)
            .WithOperand(4, m::Transpose(&transpose_arg4,
                                         m::Op(&transpose_arg4_operand)));
    if (Match(instr, pattern_arg4)) {
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Before CudnnFusedMHATransposeFusion Arg 4: \n"
                << comp->parent()->ToString();
      }
      // D_output tensor in backward graph is lhs with constraint on
      // contracting dim.
      // make sure we dont change layout of dO in flash attention case as dO
      // should have the same layout of O
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          fmha->backend_config<GpuBackendConfig>());
      const CudnnfMHABackendConfig config =
          gpu_config.cudnn_fmha_backend_config();
      if (changed && VLOG_IS_ON(2)) {
        VLOG(2) << "After CudnnFusedMHATransposeFusion Arg 4: \n"
                << comp->parent()->ToString();
      }
    }
  }
  return changed;
}

/* Let's say FMHA out is transposed to result with perm {1, 2, 0, 3} as shown
below: FMHA_out[b0, b1, n, m]{}
      |
      |
  Transpose with perm = {1, 2, 0, 3}
      |
      \/
result[b1, n, b0, m]{1, 0, 3, 2}
The goal is to find the minor_to_major of 'FMHA_out' such that it's physical
layout matches the physical layout of 'result', thus eliminating the need for an
explicit transpose. cuDNN can perform an implicit transpose by knowing the
corresponding strides (inferred from the corresponding minor_to_major).

In order to find the required mino_to_major of 'FMHA_out', we first determine
the inverse perm to obtain 'FMHA_out' from 'result'. The function
"ShapeUtil::PermuteDimensions" generates a transposed shape such that the
physical layout of the transposed shape is equivalent to the input shape.
Calling this function with 'result' shape as the input shape and the inverse
perm as the permutation will generate an output shape whose dimensions match
'FMHA_out' dimensions but the physical layout is equivalent to 'result'. This is
exactly what we want.

FMHA output should have exactly one gte instruction for a tuple index
so we can safely fuse the transpose following that gte to FMHA

FMHA_out = gte(FMHA, index=0)
FMHA_out_t = transpose(FMHA_out)
use(FMHA_out_t)

after fusion:

FMHA_out_t = gte(FMHA, index=0)
use(FMHA_out_t)
*/

absl::StatusOr<bool> FuseEpilogueTransposeWithcuDNNFMHA(HloComputation* comp) {
  bool changed = false;

  auto only_one_gte_with_spec_index = [](const HloInstruction* instr,
                                         int64_t index) {
    int count = 0;
    for (auto user : instr->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() == index) {
        count += 1;
      }
    }
    return count == 1;
  };

  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* fmha;
    HloInstruction* transpose;
    HloInstruction* gte;
    auto fwd_tuple_elem =
        m::GetTupleElement(&gte,
                           m::Op(&fmha).WithPredicate(IsFwdFMHACustomCall), 0)
            .WithOneUser();
    // Note that we don't match any specific tuple index in matcher for
    // backward.
    auto bwd_tuple_elem =
        m::GetTupleElement(&gte,
                           m::Op(&fmha).WithPredicate(IsBwdFMHACustomCall))
            .WithOneUser();
    auto fwd_pattern = m::Transpose(&transpose, fwd_tuple_elem);
    auto bwd_pattern = m::Transpose(&transpose, bwd_tuple_elem);

    if (Match(instr, fwd_pattern)) {
      // check if only one gte with such index exist
      int64_t tuple_index = gte->tuple_index();
      if (!only_one_gte_with_spec_index(fmha, tuple_index)) continue;

      std::vector<int64_t> inverse_perm =
          InversePermutation(transpose->dimensions());

      auto expected_fmha_shape =
          ShapeUtil::PermuteDimensions(inverse_perm, transpose->shape());

      // cuDNN requires the last dimension of the output to be the fastest
      // moving.
      if (expected_fmha_shape.layout().minor_to_major()[0] !=
          expected_fmha_shape.dimensions_size() - 1) {
        VLOG(3) << "cuDNN requires the last dimension of the FMHA output to be "
                   "the fastest moving. The last dimension is dim: "
                << expected_fmha_shape.dimensions_size() - 1
                << " but the upon fusion with transpose, the fmha output shape "
                   "would have been "
                << expected_fmha_shape.ToString(true)
                << " and the fastest moving "
                   "dimension would be dim: "
                << expected_fmha_shape.layout().minor_to_major()[0];
        continue;
      }
      Shape call_shape = fmha->shape();
      *call_shape.mutable_tuple_shapes(0) = expected_fmha_shape;
      HloInstruction* new_fmha_custom_call =
          comp->AddInstruction(HloInstruction::CreateCustomCall(
              call_shape, fmha->operands(),
              absl::string_view(fmha->custom_call_target())));

      TF_ASSIGN_OR_RETURN(GpuBackendConfig config,
                          fmha->backend_config<GpuBackendConfig>());
      TF_RETURN_IF_ERROR(new_fmha_custom_call->set_backend_config(config));
      TF_RETURN_IF_ERROR(
          SetFMHAInstructionName(fmha->GetModule(), new_fmha_custom_call));
      new_fmha_custom_call->set_metadata(fmha->metadata());

      auto gte = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_fmha_custom_call->shape().tuple_shapes(0), new_fmha_custom_call,
          0));
      TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
          instr, HloInstruction::CreateBitcast(transpose->shape(), gte)));
      TF_RETURN_IF_ERROR(fmha->ReplaceAllUsesWith(new_fmha_custom_call));

      if (VLOG_IS_ON(2)) {
        VLOG(2) << "After forward FuseEpilogueTransposeWithcuDNNFMHA: \n"
                << comp->parent()->ToString();
      }
      changed |= true;
    } else if (Match(instr, bwd_pattern)) {
      // check if only one gte with such index exist
      int64_t operand_tuple_idx = gte->tuple_index();
      if (!only_one_gte_with_spec_index(fmha, operand_tuple_idx)) continue;

      std::vector<int64_t> inverse_perm =
          InversePermutation(transpose->dimensions());

      auto expected_fmha_shape =
          ShapeUtil::PermuteDimensions(inverse_perm, transpose->shape());

      // cuDNN requires the last dimension of the output to be the fastest
      // moving.
      if (expected_fmha_shape.layout().minor_to_major()[0] !=
          expected_fmha_shape.dimensions_size() - 1) {
        VLOG(3) << "cuDNN requires the last dimension of the FMHA output to be "
                   "the fastest moving. The last dimension is dim: "
                << expected_fmha_shape.dimensions_size() - 1
                << " but the upon fusion with transpose, the fmha output shape "
                   "would have been "
                << expected_fmha_shape.ToString(true)
                << " and the fastest moving "
                   "dimension would be dim: "
                << expected_fmha_shape.layout().minor_to_major()[0];
        continue;
      }
      Shape call_shape = fmha->shape();
      *call_shape.mutable_tuple_shapes(operand_tuple_idx) = expected_fmha_shape;
      HloInstruction* new_fmha_custom_call =
          comp->AddInstruction(HloInstruction::CreateCustomCall(
              call_shape, fmha->operands(),
              absl::string_view(fmha->custom_call_target())));

      TF_ASSIGN_OR_RETURN(GpuBackendConfig config,
                          fmha->backend_config<GpuBackendConfig>());
      TF_RETURN_IF_ERROR(new_fmha_custom_call->set_backend_config(config));
      TF_RETURN_IF_ERROR(
          SetFMHAInstructionName(fmha->GetModule(), new_fmha_custom_call));
      new_fmha_custom_call->set_metadata(fmha->metadata());
      TF_RETURN_IF_ERROR(fmha->ReplaceAllUsesWith(new_fmha_custom_call));

      auto gte = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_fmha_custom_call->shape().tuple_shapes(operand_tuple_idx),
          new_fmha_custom_call, operand_tuple_idx));
      TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
          instr, HloInstruction::CreateBitcast(transpose->shape(), gte)));

      if (VLOG_IS_ON(2)) {
        VLOG(2) << "After backward FuseEpilogueTransposeWithcuDNNFMHA: \n"
                << comp->parent()->ToString();
      }
      changed |= true;
    }
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> CudnnFusedMHATransposeFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    bool changed = false;
    TF_ASSIGN_OR_RETURN(changed, FusePrologueTransposeWithcuDNNFMHA(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseEpilogueTransposeWithcuDNNFMHA(comp));
    any_changed |= changed;
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
