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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/evaluator/hlo_evaluator.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"

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

// Returns whether a given PrimitiveType is supported by cuBLASLt Epilogue
// Fusion. A table of supported data types can be found in the cuBLASLt
// documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul.
// Note that `Ctype` also describes the output type of the GEMM. Rows with
// `Non-default epilogue not supported` entries in the last column indicate data
// types not compatible with Epilogue Fusion.
bool SupportsEpilogueFusion(PrimitiveType type) {
  switch (type) {
    case F16:
    case BF16:
    case F32:
    case F64:
      return true;
    default:
      return false;
  }
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

auto Gemm(HloInstruction **instr) {
  return m::CustomCall(instr, kGemmCallTarget);
}

auto CublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, kCublasLtMatmulCallTarget);
}

auto GemmOrCublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget});
}

auto BcastConstScalar(HloInstruction **instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

auto BcastConstScalar(double value) { return BcastConstScalar(nullptr, value); }

auto BcastConstScalarNear(double value) {
  return m::Broadcast(m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction *instr) {
        // Not a very robust floating-point comparison, but good enough for our
        // purposes.
        std::optional<double> actual =
            static_cast<const HloConstantInstruction *>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) return false;
        double epsilon = 128 * std::numeric_limits<float>::epsilon();
        return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
      }));
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
  explicit GemmRewriterVisitor(
      se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}
  Status HandleDot(HloInstruction *instr) override {
    if (IsMatrixMultiplication(*instr)) {
      CHECK(!instr->IsRank2Transpose());
      HloInstruction *lhs = instr->mutable_operand(0);
      HloInstruction *rhs = instr->mutable_operand(1);
      CHECK(!lhs->IsRank2Transpose());
      CHECK(!rhs->IsRank2Transpose());
      const Shape &output_shape = instr->shape();

      GemmBackendConfig gemm_config;
      gemm_config.set_alpha_real(1.0);
      gemm_config.set_alpha_imag(0.0);
      gemm_config.set_beta(0.0);
      *gemm_config.mutable_dot_dimension_numbers() =
          instr->dot_dimension_numbers();
      *gemm_config.mutable_precision_config() = instr->precision_config();

      TF_ASSIGN_OR_RETURN(absl::string_view gemm_custom_call_target,
                          GetGemmCustomCallTarget(instr, gemm_config));
      std::unique_ptr<HloInstruction> gemm_call =
          HloInstruction::CreateCustomCall(output_shape, {lhs, rhs},
                                           gemm_custom_call_target);

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
                  GemmOrCublasLtMatmul(&existing_gemm).WithOneUser(),
                  m::Broadcast(m::ConstantScalar(&alpha)).WithOneUser()))) {
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
        return ReplaceInstruction(instr, existing_gemm);
      }
    }

    // Attempt to match approximate GELU activation
    // (https://arxiv.org/abs/1606.08415), where:
    // approx_gelu(x) = x * cdf(x)
    // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
    HloInstruction *cdf;
    if (Match(instr, m::MultiplyAnyOrder(CublasLtMatmul(&existing_gemm),
                                         m::Op(&cdf).WithOneUser())) &&
        Match(cdf,
              m::MultiplyAnyOrder(
                  BcastConstScalar(0.5),
                  m::AddAnyOrder(
                      BcastConstScalar(1.0),
                      m::Tanh(m::MultiplyAnyOrder(
                                  BcastConstScalarNear(sqrt(M_2_PI)),
                                  m::AddAnyOrder(
                                      m::Op().Is(existing_gemm),
                                      m::MultiplyAnyOrder(
                                          BcastConstScalarNear(0.044715),
                                          m::MultiplyAnyOrder(
                                              m::Op().Is(existing_gemm),
                                              m::MultiplyAnyOrder(
                                                  m::Op().Is(existing_gemm),
                                                  m::Op().Is(existing_gemm))
                                                  .WithOneUser())
                                              .WithOneUser())
                                          .WithOneUser())
                                      .WithOneUser())
                                  .WithOneUser())
                          .WithOneUser())))) {
      return FuseGeluActivation(instr, existing_gemm);
    }
    return OkStatus();
  }

  Status HandleAdd(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm, *optional_slice;
    // Attempt to elide broadcast and fuse addition of a vector bias into GEMM,
    // including when slicing is applied to the result.
    if (Match(instr,
              m::AddAnyOrder(m::OptionalUnaryOp(
                                 &optional_slice, {HloOpcode::kSlice},
                                 CublasLtMatmul(&existing_gemm).WithOneUser())
                                 .WithOneUser(),
                             m::Broadcast(&bias, m::Op()).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(
          bool was_fused,
          FuseVectorBiasAdd(
              instr, bias, existing_gemm,
              (optional_slice->opcode() == HloOpcode::kSlice ? optional_slice
                                                             : nullptr)));

      if (was_fused) {
        return OkStatus();
      }
    }

    // Attempt to elide broadcast and fuse addition of a vector bias into
    // *batched* GEMM as a matrix bias addition using FuseMatrixBiasAdd.
    // add(bitcast(gemm(a, b)), broadcast(bias)) ->
    //   bitcast(add(gemm(a, b), bitcast(broadcast(bias)))) ->
    //   bitcast(gemm(a, b, bitcast(broadcast(bias)))) (FuseMatrixBiasAdd)
    //
    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(CublasLtMatmul(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias, m::Op()).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_add,
          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                        MakeBitcastHlo(bias, existing_gemm->shape())));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));

      // Continue below.
      instr = new_add;
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
    // FuseMatrixBiasAdd), but if so that's okay -- we'll have done a useless
    // transformation, but it doesn't hurt anything.
    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(Gemm(&existing_gemm).WithOneUser()).WithOneUser(),
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

    if (Match(instr,
              m::AddAnyOrder(GemmOrCublasLtMatmul(&existing_gemm).WithOneUser(),
                             m::Op(&bias)))) {
      return FuseMatrixBiasAdd(instr, bias, existing_gemm);
    }

    return OkStatus();
  }

  Status HandleMaximum(HloInstruction *instr) override {
    HloInstruction *existing_gemm, *optional_slice_or_bitcast, *zeros;
    // Attempt to elide maximum and fuse ReLU activation into GEMM, including
    // when slicing or bitcasting is applied to the result.
    if (Match(instr, m::MaximumAnyOrder(
                         m::OptionalUnaryOp(
                             &optional_slice_or_bitcast,
                             {HloOpcode::kBitcast, HloOpcode::kSlice},
                             CublasLtMatmul(&existing_gemm).WithOneUser())
                             .WithOneUser(),
                         BcastConstScalar(&zeros, 0).WithOneUser()))) {
      TF_RETURN_IF_ERROR(FuseReluActivation(
          instr, zeros, existing_gemm,
          (optional_slice_or_bitcast->opcode() == HloOpcode::kSlice ||
                   optional_slice_or_bitcast->opcode() == HloOpcode::kBitcast
               ? optional_slice_or_bitcast
               : nullptr)));
    }
    return OkStatus();
  }

  Status HandleConvert(HloInstruction *instr) override {
    HloInstruction *bias, *existing_gemm;
    if (Match(instr,
              m::Convert(
                  m::AddAnyOrder(m::Convert(GemmOrCublasLtMatmul(&existing_gemm)
                                                .WithOneUser()
                                                .WithElementType(BF16))
                                     .WithOneUser(),
                                 m::Convert(m::Op(&bias).WithElementType(BF16))
                                     .WithOneUser())
                      .WithOneUser())
                  .WithElementType(BF16))) {
      return FuseMatrixBiasAdd(instr, bias, existing_gemm);
    }
    return OkStatus();
  }

  // Replaces binary(slice/bitcast(gemm), broadcast) with
  // slice/bitcast(binary(gemm, broadcast)) and changes the shape of broadcast
  // from that of slice/bitcast to that of the GEMM, i.e. the operand of
  // slice/bitcast.
  Status SinkSliceOrBitcastBelowBinaryOp(HloInstruction *slice_or_bitcast,
                                         HloInstruction **binary,
                                         HloInstruction **broadcast) {
    TF_RET_CHECK(slice_or_bitcast->user_count() == 1);
    TF_RET_CHECK((*broadcast)->user_count() == 1);

    // Re-broadcast the operand of broadcast to the shape of the GEMM.
    HloInstruction *gemm = slice_or_bitcast->mutable_operand(0);
    HloInstruction *new_broadcast = (*binary)->AddInstruction(
        (*broadcast)->CloneWithNewShape(gemm->shape()));

    // Create a new binary instruction of the same type as binary and of the
    // shape of the GEMM.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_binary,
        MakeBinaryHlo((*binary)->opcode(), gemm, new_broadcast));
    TF_RETURN_IF_ERROR(slice_or_bitcast->ReplaceOperandWith(0, new_binary));
    TF_RETURN_IF_ERROR(ReplaceInstruction(*binary, slice_or_bitcast));
    *binary = slice_or_bitcast->mutable_operand(0);
    *broadcast = new_broadcast;

    return OkStatus();
  }

  Status FuseMatrixBiasAdd(HloInstruction *instr, HloInstruction *bias,
                           const HloInstruction *gemm) {
    TF_RET_CHECK(bias->shape() == gemm->shape());

    // Do not fuse bias into S32 GEMM, as for this datatype cuBLAS only
    // supports fixed values for alpha/beta.
    if (gemm->shape().element_type() == S32) {
      return OkStatus();
    }

    // BLAS GeMM overwrites bias matrix, so fusion is only possible if the GeMM
    // is the only user. cublasLt matmul can operate out-of-place.
    bool have_other_bias_users = bias->user_count() > 1;
    bool can_fuse_bias = !have_other_bias_users || IsCublasLtMatmul(*gemm);

    auto config = gemm->backend_config<GemmBackendConfig>().value();

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
        gemm->CloneWithNewOperands(gemm->shape(), operands);

    TF_RETURN_IF_ERROR(fused_op->set_backend_config(config));

    // Choose whether the bias must alias the output. Legacy cublas GEMMs must
    // operate in place and alias the bias with the output, whereas with
    // cublasLt we can choose.
    //
    // Operating in place is always safe; copy-insertion will insert copies if
    // necessary.  But (we assume) copying is slower than operating
    // out-of-place, so for cublasLt (where we have the choice), we try to
    // operate in place if we think it a copy won't be necessary.
    //
    // We assume that parameters are always read-only and therefore we'd need to
    // copy if we were going to operate in place. (This is not quite true; the
    // param could have input/output aliasing.)  We also assume that if there
    // are other uses of the bias, we might need to copy.  (Again, not quite
    // true if those uses all come before this operation.  But copy-insertion
    // runs before scheduling, so it can't know and has to conservatively insert
    // copies.)
    if (IsLegacyCublasMatmul(*fused_op) ||
        (bias->opcode() != HloOpcode::kParameter && !have_other_bias_users)) {
      xla::Cast<HloCustomCallInstruction>(fused_op.get())
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));
    return ReplaceWithNewInstruction(instr, std::move(fused_op));
  }

  StatusOr<bool> FuseVectorBiasAdd(HloInstruction *add,
                                   HloInstruction *broadcast_bias,
                                   const HloInstruction *gemm,
                                   HloInstruction *slice = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast_bias->shape(), (slice ? slice->shape() : gemm->shape())));

    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return false;
    }

    TF_ASSIGN_OR_RETURN(auto config, gemm->backend_config<GemmBackendConfig>());

    // # output column dims == # non-contracting rhs operand dims.
    const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
    size_t num_col_dims = gemm->operand(1)->shape().rank() -
                          dot_dims.rhs_batch_dimensions_size() -
                          dot_dims.rhs_contracting_dimensions_size();

    HloInstruction *bias = broadcast_bias->mutable_operand(0);
    if ((gemm->user_count() != 1) ||
        (config.epilogue() != GemmBackendConfig::DEFAULT) ||
        (bias->shape().rank() != num_col_dims)) {
      return false;
    }
    // We require the bias vector to have been broadcast in the most major
    // dimensions; i.e. its most minor physical dimensions align with most minor
    // physical dimensions of the gemm output.
    absl::Span<const int64_t> broadcast_dims = broadcast_bias->dimensions();
    for (size_t i = 0; i < num_col_dims; ++i) {
      int64_t dim = gemm->shape().layout().minor_to_major(i);

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

    // When slicing is applied to the GEMM, replace
    // add(slice(gemm), broadcast_bias) with
    // slice(add(gemm, broadcast_bias)) to enable fusing.
    if (slice) {
      TF_RETURN_IF_ERROR(
          SinkSliceOrBitcastBelowBinaryOp(slice, &add, &broadcast_bias));
      bias = broadcast_bias->mutable_operand(0);
    }

    // Replace add(gemm, broadcast_bias) with fused new_gemm.
    config.set_epilogue(GemmBackendConfig::BIAS);

    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    operands.push_back(bias);

    std::unique_ptr<HloInstruction> new_gemm =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    TF_RETURN_IF_ERROR(new_gemm->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(add->GetModule(), new_gemm.get()));
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(add, std::move(new_gemm)));

    return true;
  }

  Status FuseReluActivation(HloInstruction *maximum,
                            HloInstruction *broadcast_zeros,
                            const HloInstruction *gemm,
                            HloInstruction *slice_or_bitcast = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast_zeros->shape(),
        (slice_or_bitcast ? slice_or_bitcast->shape() : gemm->shape())));

    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return OkStatus();
    }

    if (gemm->user_count() != 1) {
      return OkStatus();
    }

    TF_ASSIGN_OR_RETURN(auto config, gemm->backend_config<GemmBackendConfig>());
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(GemmBackendConfig::RELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(GemmBackendConfig::BIAS_RELU);
    } else {
      return OkStatus();
    }

    // When slicing or bitcasting is applied to the GEMM, replace
    // maximum(slice/bitcast(gemm), broadcast_zeros) with
    // slice/bitcast(maximum(gemm, broadcast_zeros)) to enable fusing.
    if (slice_or_bitcast) {
      TF_RETURN_IF_ERROR(SinkSliceOrBitcastBelowBinaryOp(
          slice_or_bitcast, &maximum, &broadcast_zeros));
    }

    // Replace maximum(gemm, broadcast_zeros) with fused new_gemm.
    std::unique_ptr<HloInstruction> new_gemm = gemm->Clone();
    TF_RETURN_IF_ERROR(new_gemm->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(maximum->GetModule(), new_gemm.get()));
    return ReplaceWithNewInstruction(maximum, std::move(new_gemm));
  }

  Status FuseGeluActivation(HloInstruction *multiply, HloInstruction *gemm) {
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return OkStatus();
    }

    // There are four users of the gemm output within the GELU calculation.
    bool has_aux = gemm->user_count() > 4;

    TF_ASSIGN_OR_RETURN(auto config, gemm->backend_config<GemmBackendConfig>());
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(has_aux ? GemmBackendConfig::GELU_AUX
                                  : GemmBackendConfig::GELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(has_aux ? GemmBackendConfig::BIAS_GELU_AUX
                                  : GemmBackendConfig::BIAS_GELU);
    } else {
      return OkStatus();
    }

    std::unique_ptr<HloInstruction> output = gemm->CloneWithNewShape(
        has_aux ? ShapeUtil::MakeTupleShape({gemm->shape(), gemm->shape()})
                : gemm->shape());
    TF_RETURN_IF_ERROR(output->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(multiply->GetModule(), output.get()));

    if (has_aux) {
      HloInstruction *tuple_output =
          gemm->parent()->AddInstruction(std::move(output));
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          gemm, HloInstruction::CreateGetTupleElement(tuple_output, 1)));
      output = HloInstruction::CreateGetTupleElement(tuple_output, 0);
    }

    return ReplaceWithNewInstruction(multiply, std::move(output));
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;

  StatusOr<absl::string_view> GetGemmCustomCallTarget(
      const HloInstruction *instr,
      const GemmBackendConfig &gemm_backend_config) const {
    // Decide whether or not to use cublas or cublasLt based on the instruction.
    const HloInstruction *lhs = instr->operand(0);
    const HloInstruction *rhs = instr->operand(1);
    if (!instr->GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_cublaslt()) {
      // cublasLt is not enabled.
      return absl::string_view(kGemmCallTarget);
    }

    // cublasLt is enabled.
    if (lhs->shape().element_type() == S8 ||
        rhs->shape().element_type() == S8) {
      // TODO(b/241446501) The XLA usage of cublasLt does not yet handle
      // int8 matmuls. Fallback to legacy cublas.
      return absl::string_view(kGemmCallTarget);
    }

    TF_ASSIGN_OR_RETURN(bool gemm_is_supported_by_cublas_lt,
                        GemmIsSupportedByCublasLt(instr, gemm_backend_config));
    if (gemm_is_supported_by_cublas_lt) {
      return absl::string_view(kCublasLtMatmulCallTarget);
    }

    // This case is not supported by cublasLt, fallback to legacy cublas.
    return absl::string_view(kGemmCallTarget);
  }

  StatusOr<bool> TypesAreSupportedByCublasLt(
      const HloInstruction *instr) const {
    // cublasLt has a defined set of combinations of types that it supports.
    // Figure out the computeType and scaleType.
    TF_ASSIGN_OR_RETURN(const se::blas::DataType output_dtype,
                        AsBlasDataType(instr->shape().element_type()));
    TF_ASSIGN_OR_RETURN(const se::blas::ComputationType compute_type,
                        GetBlasComputationType(instr->shape().element_type()));
    se::blas::DataType scale_type =
        cublas_lt::GetScaleType(output_dtype, compute_type);

    // Figure out the Atype/Btype.
    const PrimitiveType a_dtype = instr->operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr->operand(1)->shape().element_type();

    if (a_dtype != b_dtype) {
      // AType must match BType.
      return false;
    }

    using se::blas::ComputationType;
    using se::blas::DataType;
    // This matrix of supported types is taken directly from cublasLt
    // documentation.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    const std::array<
        std::tuple<ComputationType, DataType /*scale_type*/,
                   PrimitiveType /*a_dtype*/, DataType /*output_dtype*/>,
        18>
        supported_type_combinations = {{
            {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
             DataType::kHalf},

            {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
             DataType::kInt32},
            {ComputationType::kI32, DataType::kFloat, PrimitiveType::S8,
             DataType::kInt8},

            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
             DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
             DataType::kFloat},

            // There would be an entry here for A/BType complex int8, but we do
            // not support that type.
            {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
             DataType::kComplexFloat},

            {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
             DataType::kFloat},
            {ComputationType::kF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
             DataType::kFloat},
            {ComputationType::kBF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
             DataType::kFloat},
            {ComputationType::kTF32AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
             DataType::kDouble},
            {ComputationType::kF64, DataType::kComplexDouble,
             PrimitiveType::C128, DataType::kComplexDouble},
        }};

    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, output_dtype));
  }

  StatusOr<bool> GemmIsSupportedByCublasLt(
      const HloInstruction *instr,
      const GemmBackendConfig &gemm_backend_config) const {
    const HloInstruction *lhs = instr->operand(0);
    const HloInstruction *rhs = instr->operand(1);
    const Shape &output_shape = instr->shape();

    TF_ASSIGN_OR_RETURN(bool types_are_supported_by_cublas_lt,
                        TypesAreSupportedByCublasLt(instr));
    if (!types_are_supported_by_cublas_lt) {
      return false;
    }

    // The cublasLt API has two currently known limitations:
    // 1. Batch count must be <2^16.
    constexpr int64_t kMaxBatchCount = 65535;
    // We get the batch dimension size from lhs here, but we could just as well
    // use rhs; they are guaranteed to be the same (TODO:Verify).
    const auto &batch_dimensions =
        gemm_backend_config.dot_dimension_numbers().lhs_batch_dimensions();
    int batch_count = (batch_dimensions.empty() ? 0 : 1);
    // All batch dimensions get flattened into a single batch dimension.
    for (auto batch_dimension : batch_dimensions) {
      batch_count *= lhs->shape().dimensions(batch_dimension);
    }
    if (batch_count > kMaxBatchCount) {
      // This is not supported by cublasLt.
      return false;
    }

    // 2. cublasLt does not support rhs col dimension size > 4194240 for
    // C64.
    constexpr int kMaxDimensionSize{4194240};
    if (output_shape.element_type() != C64) {
      // Does not match type in unsupported case.
      return true;
    }

    if (cuda_compute_capability_.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
      // cuBlasLt has an implementation for complex data with compute type
      // 32F_FAST_32TF that uses tensor cores and that is free from the
      // restriction. This implementation only works on Ampere
      // architecture though (where TF32 was introduced).
      return true;
    }

    // Get the rhs non-contracting dimensions as they will eventually be at the
    // cublasLt level.
    std::vector<int64_t> rhs_non_contracting_dims;
    const DotDimensionNumbers &dot_dims =
        gemm_backend_config.dot_dimension_numbers();
    TF_ASSIGN_OR_RETURN(
        GemmConfig gemm_config,
        GemmConfig::For(
            lhs->shape(), dot_dims.lhs_batch_dimensions(),
            dot_dims.lhs_contracting_dimensions(), rhs->shape(),
            dot_dims.rhs_batch_dimensions(),
            dot_dims.rhs_contracting_dimensions(),
            /*output_shape=*/instr->shape(), gemm_backend_config.alpha_real(),
            gemm_backend_config.alpha_imag(), gemm_backend_config.beta(),
            /*algorithm*/ std::nullopt, se::blas::kDefaultComputePrecision));
    if (gemm_config.output_layout.order != MatrixLayout::Order::kColumnMajor) {
      // cublasLt's matmul output is column major by default. This gemm requires
      // the output to be in row major. Later we will swap lhs & rhs (and
      // transpose each operand) of this gemm. Since we care about the rhs at
      // the cublasLt level, this swap means that we care about the lhs right
      // here.
      TF_ASSIGN_OR_RETURN(
          rhs_non_contracting_dims,
          GetNonContractingDims(lhs->shape(), dot_dims.lhs_batch_dimensions(),
                                dot_dims.lhs_contracting_dimensions()));
    } else {
      TF_ASSIGN_OR_RETURN(
          rhs_non_contracting_dims,
          GetNonContractingDims(rhs->shape(), dot_dims.rhs_batch_dimensions(),
                                dot_dims.rhs_contracting_dimensions()));
    }

    const auto lhs_non_contracting_dimension_size = absl::c_accumulate(
        rhs_non_contracting_dims, 1, [&](int64_t size, int64_t dim) {
          return size * lhs->shape().dimensions(dim);
        });

    // Check that the size of the non-contracting dimension is not too large.
    return lhs_non_contracting_dimension_size <= kMaxDimensionSize;
  }
};

StatusOr<bool> RunOnComputation(
    HloComputation *computation,
    se::CudaComputeCapability cuda_compute_capability) {
  GemmRewriterVisitor visitor(cuda_compute_capability);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

GemmRewriter::GemmRewriter(se::CudaComputeCapability cuda_compute_capability)
    : cuda_compute_capability_(cuda_compute_capability) {}

StatusOr<bool> GemmRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
