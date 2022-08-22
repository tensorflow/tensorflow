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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

// Give this instruction a more useful name than "custom-call.42".
Status SetName(HloModule *module, HloInstruction *gemm) {
  if (IsCublasLtMatmul(*gemm)) {
    module->SetAndUniquifyInstrName(gemm, "cublas-lt-matmul");
    return OkStatus();
  }

  GemmBackendConfig config;
  TF_ASSIGN_OR_RETURN(config, gemm->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();

  module->SetAndUniquifyInstrName(
      gemm, is_batch_dot ? "cublas-batch-gemm" : "cublas-gemm");
  return OkStatus();
}

// If the bias is a sequence of ops that depend only on broadcasts of
// constants, materialize the bias if it's small.
//
// Normally the constant-folding pass would materialize the bias if it is
// calculated entirely from constants. But if the bias is a broadcast of a
// constant, constant-folding won't expand the broadcast, on the theory that
// folding broadcasts of constants causes us to consume more memory and can
// actually make things slower (because any op which reads the constant has
// to read more memory).
//
// OTOH in our case, we don't want to run an op that just broadcasts a
// constant so we can fuse it into this gemm. That would defeat the whole
// purpose of this fusion, which is to launch fewer kernels.  So if we can,
// we expand out this constant ourselves.
//
// TODO(b/192499646): Even better would be to use cublasLT to fuse the
// broadcasted bias, if it supports that fusion efficiently.
HloInstruction *MaybeConstantFoldBias(HloInstruction *bias) {
  // This limit was not chosen carefully.
  constexpr int kMaxMaterializeBiasBytes = 8 * 1024 * 1024;

  // Don't fold broadcasts of scalars -- algsimp will just collapse it again.
  auto is_nonscalar = [](const HloInstruction *instr) {
    return !ShapeUtil::IsEffectiveScalar(instr->shape());
  };

  // For now, only fold broadcast(constant) or
  // reshape/transpose/bitcast(broadcast(constant)). This lets us avoid the
  // complexity in the constant-folding pass about what is and isn't legal to
  // fold.
  auto broadcast_of_nonscalar =
      m::Broadcast(m::Constant().WithPredicate(is_nonscalar));

  if (ShapeUtil::ByteSizeOf(bias->shape()) <= kMaxMaterializeBiasBytes &&
      (Match(bias, broadcast_of_nonscalar) ||
       Match(bias, m::Reshape(broadcast_of_nonscalar)) ||
       Match(bias, m::Transpose(broadcast_of_nonscalar)) ||
       Match(bias, m::Bitcast(broadcast_of_nonscalar)))) {
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    Literal result;
    if (evaluator.TryEvaluate(
            bias, &result,
            /*recursively_evaluate_nonconstant_operands=*/true)) {
      return bias->parent()->AddInstruction(
          HloInstruction::CreateConstant(std::move(result)));
    }
  }

  return bias;
}

// The rewriting proceeds in a bottom-up way:
//
// (kDot A B) is rewritten into a (kCustomCall:gemm A B)
//
// (kMultiply (kCustomCall:gemm A B) C) is folding C (provided it's a constant)
// into an alpha parameter of the custom call.
//
// (kAdd (kCustomCall:gemm A B) C) is rewritten into (kCustomCall:gemm A B C),
// where the "beta" parameter is set to 1 (provided it was zero before,
// and provided C has no other users).
// We then guide the buffer assignment to alias the buffer of the custom call
// and C.
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleDot(HloInstruction *instr) override {
    if (IsMatrixMultiplication(*instr)) {
      CHECK(!instr->IsRank2Transpose());
      HloInstruction *lhs = instr->mutable_operand(0);
      HloInstruction *rhs = instr->mutable_operand(1);
      CHECK(!lhs->IsRank2Transpose());
      CHECK(!rhs->IsRank2Transpose());
      const Shape &output_shape = instr->shape();

      const char *const target =
          instr->GetModule()->config().debug_options().xla_gpu_enable_cublaslt()
              ? kCublasLtMatmulCallTarget
              : kGemmCallTarget;

      std::unique_ptr<HloInstruction> gemm_call =
          HloInstruction::CreateCustomCall(output_shape, {lhs, rhs}, target);
      GemmBackendConfig gemm_config;
      gemm_config.set_alpha_real(1.0);
      gemm_config.set_alpha_imag(0.0);
      gemm_config.set_beta(0.0);
      *gemm_config.mutable_dot_dimension_numbers() =
          instr->dot_dimension_numbers();
      *gemm_config.mutable_precision_config() = instr->precision_config();

      TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gemm_config));
      TF_RETURN_IF_ERROR(SetName(instr->GetModule(), gemm_call.get()));
      TF_RETURN_IF_ERROR(
          ReplaceWithNewInstruction(instr, std::move(gemm_call)));
    }
    return OkStatus();
  }

  Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  m::Op(&existing_gemm).WithCustomCallTarget(kGemmCallTarget),
                  m::Broadcast(m::ConstantScalar(&alpha))))) {
      TF_ASSIGN_OR_RETURN(auto config,
                          existing_gemm->backend_config<GemmBackendConfig>());

      // Do not fuse alpha into S32 GEMM, as they only support fixed values for
      // alpha/beta.
      if (existing_gemm->shape().element_type() == S32) {
        return OkStatus();
      }

      if (config.beta() == 0.0 && existing_gemm->user_count() == 1) {
        complex128 prev_alpha = {config.alpha_real(), config.alpha_imag()};
        complex128 new_alpha =
            *alpha->literal().GetAsComplex128({}) * prev_alpha;
        config.set_alpha_real(new_alpha.real());
        config.set_alpha_imag(new_alpha.imag());
        TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(config));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, existing_gemm));
      }
    }
    return OkStatus();
  }

  Status HandleAdd(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm;

    // First, try to match vector bias add, so we might elide the broadcast.
    if (Match(instr, m::AddAnyOrder(
                         m::Op(&existing_gemm)
                             .WithCustomCallTarget(kCublasLtMatmulCallTarget),
                         m::Broadcast(&bias, m::Op())))) {
      TF_ASSIGN_OR_RETURN(bool was_fused,
                          FuseVectorBiasAdd(instr, bias, existing_gemm));
      if (was_fused) return OkStatus();
    }

    // add(bitcast(gemm(a, b)), bias) ->
    //   bitcast(add(gemm(a, b), bitcast(bias))) ->
    //   bitcast(gemm(a, b, bitcast(bias))) (later down in this function).
    //
    // We see this idiom in models that contain batch-dots, where we cast
    // between a rank-2 shape for non-batch dots and a higher-rank shape for
    // batch-dots.
    //
    // The last stage of the transform may fail (because of any of the checks in
    // FuseBiasedGemm), but if so that's okay -- we'll have done a useless
    // transformation, but it doesn't hurt anything.
    if (Match(instr, m::AddAnyOrder(
                         m::Bitcast(m::Op(&existing_gemm)
                                        .WithCustomCallTarget(kGemmCallTarget)
                                        .WithOneUser())
                             .WithOneUser(),
                         m::Op(&bias)))) {
      HloInstruction *new_bitcast =
          MakeBitcastHlo(bias, existing_gemm->shape(), &bias->metadata());
      TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                                        new_bitcast, &bias->metadata()));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));

      // Continue below transforming new_add.
      instr = new_add;
    }

    if (Match(instr, m::AddAnyOrder(
                         m::Op(&existing_gemm)
                             .WithCustomCallTarget(
                                 {kGemmCallTarget, kCublasLtMatmulCallTarget}),
                         m::Op(&bias)))) {
      return FuseMatrixBiasAdd(instr, bias, existing_gemm);
    }

    return Status::OK();
  }

  Status HandleConvert(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm;
    if (Match(
            instr,
            m::Convert(m::AddAnyOrder(
                           m::Convert(m::Op(&existing_gemm)
                                          .WithCustomCallTarget(kGemmCallTarget)
                                          .WithElementType(BF16)),
                           m::Convert(m::Op(&bias).WithElementType(BF16))))
                .WithElementType(BF16))) {
      return FuseMatrixBiasAdd(instr, bias, existing_gemm);
    }
    return OkStatus();
  }

  Status FuseMatrixBiasAdd(HloInstruction *instr, HloInstruction *bias,
                           HloInstruction *gemm) {
    TF_RET_CHECK(bias->shape() == gemm->shape());

    // Do not fuse bias into S32 GEMM, as for this datatype cuBLAS only
    // supports fixed values for alpha/beta.
    if (gemm->shape().element_type() == S32) {
      return OkStatus();
    }

    // BLAS GeMM overwrites bias matrix, so fusion is only possible if the GeMM
    // is the only user. cublasLt matmul can operate out-of-place.
    bool can_fuse_bias = (bias->user_count() == 1) || IsCublasLtMatmul(*gemm);

    auto config = gemm->backend_config<GemmBackendConfig>().ValueOrDie();

    // It is possible to fuse into a cublasLt matmul that already has a vector
    // bias, but no other epilogue will commute with the matrix bias add.
    bool supported_epilogue =
        ((config.epilogue() == GemmBackendConfig::DEFAULT) ||
         (config.epilogue() == GemmBackendConfig::BIAS));

    if ((config.beta() != 0) || !can_fuse_bias || (gemm->user_count() != 1) ||
        !supported_epilogue) {
      return OkStatus();
    }

    config.set_beta(1.0);

    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    operands.insert(operands.begin() + 2, MaybeConstantFoldBias(bias));

    std::unique_ptr<HloInstruction> fused_op =
        gemm->CloneWithNewOperands(instr->shape(), operands);

    TF_RETURN_IF_ERROR(fused_op->set_backend_config(config));
    if (IsCublasGemm(*fused_op)) {
      // Force bias input to alias with output, as GEMM operates in-place.
      xla::Cast<HloCustomCallInstruction>(fused_op.get())
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(fused_op)));
    return OkStatus();
  }

  StatusOr<bool> FuseVectorBiasAdd(HloInstruction *instr,
                                   HloInstruction *broadcast_bias,
                                   HloInstruction *matmul) {
    TF_RET_CHECK(broadcast_bias->shape() == matmul->shape());

    auto config = matmul->backend_config<GemmBackendConfig>().ValueOrDie();

    // # output column dims == # non-contracting rhs operand dims.
    const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
    size_t num_col_dims = matmul->operand(1)->shape().rank() -
                          dot_dims.rhs_batch_dimensions_size() -
                          dot_dims.rhs_contracting_dimensions_size();

    HloInstruction *bias = broadcast_bias->mutable_operand(0);
    if ((matmul->user_count() != 1) ||
        (config.epilogue() != GemmBackendConfig::DEFAULT) ||
        (bias->shape().rank() != num_col_dims)) {
      return false;
    }

    // We require the bias vector to have been broadcast in the most major
    // dimensions; i.e. its most minor physical dimensions align with most minor
    // physical dimensions of the matmul output.
    absl::Span<const int64_t> broadcast_dims = broadcast_bias->dimensions();
    for (size_t i = 0; i < num_col_dims; ++i) {
      int64_t dim = matmul->shape().layout().minor_to_major(i);

      // Find the corresponding dimension from the bias vector.
      auto it = absl::c_find(broadcast_dims, dim);

      if (it == broadcast_dims.end()) {
        return false;
      }

      int64_t vector_dim = it - broadcast_dims.begin();
      if (bias->shape().layout().minor_to_major(i) != vector_dim) {
        return false;
      }
    }

    std::vector<HloInstruction *> operands(matmul->operands().begin(),
                                           matmul->operands().end());
    operands.push_back(bias);

    std::unique_ptr<HloInstruction> fused_op =
        matmul->CloneWithNewOperands(instr->shape(), operands);

    config.set_epilogue(GemmBackendConfig::BIAS);
    TF_RETURN_IF_ERROR(fused_op->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(fused_op)));
    return true;
  }
};

StatusOr<bool> RunOnComputation(HloComputation *computation) {
  GemmRewriterVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

StatusOr<bool> GemmRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
