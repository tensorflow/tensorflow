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
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/dnn.pb.h"

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
    case F8E4M3FN:
    case F8E5M2:
    case F16:
    case BF16:
    case F32:
    case F64:
      return true;
    default:
      return false;
  }
}

bool IsF8Type(const HloInstruction *instr) {
  return primitive_util::IsF8Type(instr->shape().element_type());
}

// Recursively collects unary, pad, divide or multiply operands of instr until
// an instruction with FP8 element type is reached. Returns std::nullopt when no
// FP8 instruction is reached.
std::optional<std::vector<HloInstruction *>> FindF8SubgraphRecursive(
    HloInstruction *instr, absl::flat_hash_set<int> &visited_instrs,
    std::vector<HloInstruction *> subgraph) {
  // Avoid visiting the same instruction more than once.
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return std::nullopt;
  }
  subgraph.emplace_back(instr);
  if (IsF8Type(instr)) {
    return subgraph;
  } else {
    if (instr->operand_count() == 1 || instr->opcode() == HloOpcode::kDivide ||
        instr->opcode() == HloOpcode::kPad) {
      return FindF8SubgraphRecursive(instr->mutable_operand(0), visited_instrs,
                                     subgraph);
    } else if (instr->opcode() == HloOpcode::kMultiply) {
      for (int k = 0; k < 2; ++k) {
        auto mult_subgraph = FindF8SubgraphRecursive(instr->mutable_operand(k),
                                                     visited_instrs, subgraph);
        if (mult_subgraph.has_value()) {
          return mult_subgraph;
        }
      }
    }
    return std::nullopt;
  }
}

// Returns whether instr and its operands describe a pattern which is compatible
// with rewriting the dot operating on instr into an FP8 Custom Call. If
// applicable, captures the operand of the Custom Call, its scaling factor,
// whether the scaling factor is applied by multiplication and intermediate
// unary ops.
bool IsSupportedF8Pattern(HloInstruction *instr, HloInstruction *&x,
                          HloInstruction *&x_scale, bool &x_mult_scale,
                          std::vector<HloInstruction *> &x_unary_ops) {
  absl::flat_hash_set<int> visited_instrs;
  std::optional<std::vector<HloInstruction *>> subgraph =
      FindF8SubgraphRecursive(instr, visited_instrs,
                              std::vector<HloInstruction *>{});

  if (!subgraph.has_value()) {
    return false;
  }
  std::reverse(subgraph->begin(), subgraph->end());

  // Directly operating on an FP8 operand.
  if (subgraph->size() == 1) {
    x = (*subgraph)[0];
    return true;
  }

  // When not operating directly on an FP8 operand, the second and
  // third instructions in the subgraph must describe a dequantization, i.e. a
  // convert instruction followed by a multiply/divide instruction.
  if (subgraph->size() > 2 &&
      Match((*subgraph)[2],
            m::MultiplyAnyOrder(m::Convert(m::Op(&x)),
                                m::Broadcast(m::Op(&x_scale))))) {
    x_mult_scale = true;
  } else if (subgraph->size() > 2 &&
             Match((*subgraph)[2], m::Divide(m::Convert(m::Op(&x)),
                                             m::Broadcast(m::Op(&x_scale))))) {
    x_mult_scale = false;
  } else {
    VLOG(1) << "Possible intended FP8 GEMM operating on "
            << instr->ToShortString() << " not rewritten into FP8 Custom Call.";
    return false;
  }

  auto preserves_element_type = [](const HloInstruction *instr) -> bool {
    return ShapeUtil::SameElementType(instr->shape(),
                                      instr->operand(0)->shape());
  };
  for (int i = 3; i < subgraph->size(); ++i) {
    // The remaining instructions must be commutative with dequantization.
    // Bitcast, broadcast, copy, pad, reshape and slice instructions are
    // supported.
    if (!Match((*subgraph)[i],
               m::AnyOf<HloInstruction>(
                   m::Bitcast().WithPredicate(preserves_element_type),
                   m::Broadcast(), m::Copy(), m::Pad(), m::Reshape(),
                   m::Slice()))) {
      VLOG(1) << "Possible intended FP8 GEMM operating on "
              << instr->ToShortString()
              << " not rewritten into FP8 Custom Call.";
      return false;
    }
  }

  x_unary_ops = {subgraph->begin() + 3, subgraph->end()};
  return true;
}

// Transposes a matrix by swapping the contracting and non-contracting
// dimension. There must be only one contracting and only one non-contracting
// dimension. Keeps the layout the same.
HloInstruction *TransposeMatrix(HloInstruction *instr, int64_t contracting_dim,
                                absl::Span<const int64_t> batch_dims) {
  // Identify the dimensional order which describes a transpose of the
  // contracting and non-contracting dimensions of the GEMM.
  std::vector<int64_t> permutation(instr->shape().dimensions_size(), -1);
  // Discard the batch dimensions.
  for (int64_t batch_dim : batch_dims) {
    permutation[batch_dim] = batch_dim;
  }
  // Identify the non-contracting dimension.
  int non_contracting_dim;
  for (int i = 0; i < instr->shape().dimensions_size(); ++i) {
    if (permutation[i] == -1 && contracting_dim != i) {
      non_contracting_dim = i;
    }
  }
  permutation[non_contracting_dim] = contracting_dim;
  permutation[contracting_dim] = non_contracting_dim;

  Shape new_shape = ShapeUtil::PermuteDimensions(permutation, instr->shape());
  *new_shape.mutable_layout() = instr->shape().layout();
  return instr->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, instr, permutation));
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
  return m::CustomCall(instr, {kGemmCallTarget});
}

auto CublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kCublasLtMatmulCallTarget});
}

auto GemmOrCublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget});
}

auto CublasLtMatmulMaybeF8(HloInstruction **instr) {
  return m::CustomCall(
      instr, {kCublasLtMatmulCallTarget, kCublasLtMatmulF8CallTarget});
}

auto GemmOrCublasLtMatmulMaybeF8(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget,
                               kCublasLtMatmulF8CallTarget});
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

template <typename Pattern>
auto OptionalSlice(HloInstruction **optional_slice, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Slice(optional_slice, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalConvert(HloInstruction **optional_convert, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(optional_convert, pattern),
                                  std::move(pattern));
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
//
// For scaled FP8 GEMMs on Hopper systems, the following steps mentioned in
// RFC #22 (https://github.com/openxla/xla/discussions/22) are elided and
// rewritten into a Custom Call:
//
// 1. Cast each input from FP8 to a wider type such as FP16 or FP32.
// 2. Unscale each input by multiplying each input by the corresponding input
// scale.
// 3. Evaluate the matrix multiplication on the scaled inputs.
// 4. Compute the maximum of the absolute values in the result of the GEMM
// (DAmax).
// 5. Scale the output by dividing the output by the output scale.
// 6. Cast the output back to FP8. Since saturation should be done on
// overflow, this is represented by a Clamp instruction followed by a Convert
// instruction.

// Steps 1 through 3 can be elided independently of the remainder. Steps 5 and
// 6 are elided only if steps 1 through 3 were successfully transformed. Step
// 4 requires steps 5 and 6, i.e. the computation of DAmax can be elided only
// when the output of the GEMM is requested in FP8 format.
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterVisitor(GpuVersion gpu_version)
      : gpu_version_(gpu_version) {}

  Status HandleDot(HloInstruction *instr) override {
    if (!IsMatrixMultiplication(*instr)) {
      return OkStatus();
    }

    CHECK(!instr->IsRank2Transpose());
    CHECK(!instr->mutable_operand(0)->IsRank2Transpose());
    CHECK(!instr->mutable_operand(1)->IsRank2Transpose());

    // Create a GemmBackendConfig based on the instruction.
    GemmBackendConfig gemm_backend_config;
    gemm_backend_config.set_alpha_real(1.0);
    gemm_backend_config.set_alpha_imag(0.0);
    gemm_backend_config.set_beta(0.0);
    *gemm_backend_config.mutable_dot_dimension_numbers() =
        instr->dot_dimension_numbers();
    *gemm_backend_config.mutable_precision_config() = instr->precision_config();

    // First try to match the fp8 gemm pattern.
    TF_ASSIGN_OR_RETURN(bool supported_by_cublaslt,
                        GemmIsSupportedByCublasLt(*instr, gemm_backend_config));
    HloInstruction *a, *b, *a_scale = nullptr, *b_scale = nullptr;
    std::vector<HloInstruction *> a_unary_ops, b_unary_ops;
    bool a_mult_scale, b_mult_scale;
    if (supported_by_cublaslt &&
        Match(instr,
              m::Dot(m::Op().WithPredicate([&](const HloInstruction *instr) {
                return IsSupportedF8Pattern(const_cast<HloInstruction *>(instr),
                                            a, a_scale, a_mult_scale,
                                            a_unary_ops);
              }),
                     m::Op().WithPredicate([&](const HloInstruction *instr) {
                       return IsSupportedF8Pattern(
                           const_cast<HloInstruction *>(instr), b, b_scale,
                           b_mult_scale, b_unary_ops);
                     })))) {
      TF_ASSIGN_OR_RETURN(
          bool created_call,
          CreateF8CustomCall(instr, gemm_backend_config, a, b, a_scale, b_scale,
                             a_mult_scale, b_mult_scale, a_unary_ops,
                             b_unary_ops));
      if (created_call) {
        return OkStatus();
      }
    }

    if (IsF8Type(instr->operand(0))) {
      // Couldn't rewrite as an FP8 cublasLt custom call, so turn into an FP16
      // dot and below it will be rewritten as an FP16 cublas or cublasLt call.
      TF_ASSIGN_OR_RETURN(instr, TurnF8DotIntoF16Dot(instr));
    }

    // Couldn't rewrite as an FP8 cublasLt custom call, rewrite as a cublas or
    // cublasLt call.
    TF_ASSIGN_OR_RETURN(
        absl::string_view gemm_custom_call_target,
        GetNonFp8GemmCustomCallTarget(*instr, gemm_backend_config));
    const Shape &output_shape = instr->shape();
    HloInstruction *gemm_call =
        instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape,
            {instr->mutable_operand(0), instr->mutable_operand(1)},
            gemm_custom_call_target));
    TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gemm_backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, gemm_call));
    return OkStatus();
  }

  Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  GemmOrCublasLtMatmulMaybeF8(&existing_gemm).WithOneUser(),
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
    HloInstruction *bias, *existing_gemm;
    HloInstruction *optional_slice = nullptr;
    HloInstruction *optional_convert = nullptr;
    // Attempt to elide broadcast and fuse addition of a vector bias into GEMM,
    // including when slicing is applied to the result.
    if (Match(instr,
              m::AddAnyOrder(
                  OptionalSlice(
                      &optional_slice,
                      CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias,
                               OptionalConvert(&optional_convert, m::Op()))))) {
      TF_ASSIGN_OR_RETURN(bool was_fused,
                          FuseVectorBiasAdd(instr, bias, existing_gemm,
                                            optional_slice, optional_convert));

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

    // Do not fuse broadcast unless we can fuse its input, as it will cause
    // broadcast materialization.
    auto is_not_broadcast = [](const HloInstruction *instr) {
      return instr->opcode() != HloOpcode::kBroadcast;
    };

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
                  m::Bitcast(GemmOrCublasLtMatmul(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
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
                             m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      return FuseMatrixBiasAdd(instr, bias, existing_gemm);
    }

    return OkStatus();
  }

  Status HandleMaximum(HloInstruction *instr) override {
    HloInstruction *existing_gemm, *zeros;
    HloInstruction *optional_slice_or_bitcast = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM, including
    // when slicing or bitcasting is applied to the result.
    if (Match(instr,
              m::MaximumAnyOrder(
                  m::AnyOf<HloInstruction>(
                      m::Slice(
                          &optional_slice_or_bitcast,
                          CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser()),
                      m::Bitcast(
                          &optional_slice_or_bitcast,
                          CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser()),
                      CublasLtMatmulMaybeF8(&existing_gemm))
                      .WithOneUser(),
                  m::Broadcast(&zeros, m::ConstantScalar(0))))) {
      TF_RETURN_IF_ERROR(FuseReluActivation(instr, zeros, existing_gemm,
                                            optional_slice_or_bitcast));
    }
    return OkStatus();
  }

  Status HandleConvert(HloInstruction *instr) override {
    HloInstruction *clamp_lower, *clamp_upper, *d_scale, *existing_gemm,
        *binary;

    // Attempt to elide the scaling and conversion of the result of an FP8
    // GEMM, including the optional calculation of the maximum of the absolute
    // values before scaling, and adapt the Custom Call.
    if (Match(instr,
              m::Convert(
                  m::Clamp(m::Broadcast(m::ConstantScalar(&clamp_lower)),
                           m::AnyOf<HloInstruction>(
                               m::Divide(
                                   &binary,
                                   m::CustomCall(&existing_gemm,
                                                 {kCublasLtMatmulF8CallTarget}),
                                   m::Broadcast(m::Op(&d_scale))),
                               m::MultiplyAnyOrder(
                                   &binary,
                                   m::CustomCall(&existing_gemm,
                                                 {kCublasLtMatmulF8CallTarget}),
                                   m::Broadcast(m::Op(&d_scale)))),
                           m::Broadcast(m::ConstantScalar(&clamp_upper)))
                      .WithOneUser()))) {
      return F8ConvertD(
          instr, existing_gemm, d_scale, clamp_lower, clamp_upper,
          /*mult_scale=*/binary->opcode() == HloOpcode::kMultiply);
    }
    return OkStatus();
  }

  StatusOr<bool> CreateF8CustomCall(HloInstruction *instr,
                                    GemmBackendConfig &gemm_backend_config,
                                    HloInstruction *a, HloInstruction *b,
                                    HloInstruction *a_scale,
                                    HloInstruction *b_scale, bool a_mult_scale,
                                    bool b_mult_scale,
                                    std::vector<HloInstruction *> a_unary_ops,
                                    std::vector<HloInstruction *> b_unary_ops) {
    auto cuda_compute_capability_ =
        std::get<se::CudaComputeCapability>(gpu_version_);
    // FP8 GEMM kernels are only available on Hopper and newer architectures.
    if (!cuda_compute_capability_.IsAtLeast(
            se::CudaComputeCapability::HOPPER)) {
      VLOG(1) << "FP8 Custom Calls require Hopper or newer architecture.";
      return false;
    }
#if CUDA_VERSION < 12000
    // FP8 GEMM kernels are only available with CUDA 12.0 and above
    VLOG(1) << "FP8 Custom Calls require CUDA 12.0 or newer.";
    return false;
#endif  // CUDA_VERSION < 12000

    // cuBLASLt FP8 GEMM kernels require one of the two operands to be in
    // F8E4M3FN format.
    if (a->shape().element_type() == F8E5M2 &&
        b->shape().element_type() == F8E5M2) {
      VLOG(1)
          << "Failed to rewrite " << instr->ToShortString()
          << " into FP8 Custom Call. The element type of one of the operands "
             "must be F8E4M3FN.";
      return false;
    }

    absl::Span<const int64_t> batch_dims =
        gemm_backend_config.dot_dimension_numbers().rhs_batch_dimensions();

    // cuBLASLt FP8 GEMM kernels require the scaling factors to be in F32
    // format. Set the factors to one when no scaling factors were captured.
    Literal one_literal = LiteralUtil::One(F32);
    HloInstruction *one = instr->AddInstruction(
        HloInstruction::CreateConstant(one_literal.Clone()));
    std::array<bool, 2> mult_scale{a_mult_scale, b_mult_scale};
    std::array<HloInstruction *, 2> scales{a_scale, b_scale}, inv_scales,
        scales_f32;
    for (int i = 0; i < scales.size(); ++i) {
      if (scales[i]) {
        if (!ShapeUtil::IsScalar(scales[i]->shape())) {
          VLOG(1) << "Failed to rewrite " << instr->ToShortString()
                  << " into FP8 Custom Call. The scaling factors must be "
                     "scalars.";
          return false;
        }
        if (!mult_scale[i]) {
          inv_scales[i] = instr->AddInstruction(HloInstruction::CreateBinary(
              scales[i]->shape(), HloOpcode::kDivide, one, scales[i]));
        }
        scales_f32[i] = mult_scale[i] ? scales[i] : inv_scales[i];
        if (scales_f32[i]->shape().element_type() != F32) {
          scales_f32[i] = instr->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::MakeScalarShape(F32), scales_f32[i]));
        }
      } else {
        scales_f32[i] = one;
      }
    }

    PrimitiveType c_type;
    switch (instr->shape().element_type()) {
      case F8E4M3FN:
      case F8E5M2:
      case BF16:
        c_type = BF16;
        break;
      case F16:
        c_type = F16;
        break;
      case F32:
        c_type = F32;
        break;
      default:
        VLOG(1) << "Failed to rewrite " << instr->ToShortString()
                << " into FP8 Custom Call. Output element type must be "
                   "F8E4M3FN, F8E5M2, BF16, F16 or F32. Actual element type is "
                << PrimitiveType_Name(instr->shape().element_type());
        return false;
    }

    // Fuse the possible addition of a matrix bias here to enable the subsequent
    // fusion of the scaling and conversion of D into the Custom Call. Fusing
    // a matrix bias is only supported with CUDA 12 and above.
    HloInstruction *c = nullptr;
    if (instr->user_count() == 1 &&
        instr->users()[0]->opcode() == HloOpcode::kAdd) {
      HloInstruction *add = instr->users()[0];
      HloInstruction *bias = add->mutable_operand(!add->operand_index(instr));
      if (bias->opcode() != HloOpcode::kBroadcast) {
        c = bias;
        gemm_backend_config.set_beta(1.0);
        TF_RETURN_IF_ERROR(ReplaceInstruction(add, instr));
      }
    }

    // Each operand must have exactly one contracting and one non-contracting
    // dimension.
    absl::Span<const int64_t> a_contracting_dims =
        gemm_backend_config.dot_dimension_numbers()
            .lhs_contracting_dimensions();
    absl::Span<const int64_t> b_contracting_dims =
        gemm_backend_config.dot_dimension_numbers()
            .rhs_contracting_dimensions();
    if (a_contracting_dims.size() != 1 || b_contracting_dims.size() != 1) {
      VLOG(1) << "Failed to rewrite " << instr->ToShortString()
              << " into FP8 Custom Call. A and B must have one contracting "
                 "dimension.";
      return false;
    }
    if ((a_unary_ops.empty() ? a : a_unary_ops.back())
                    ->shape()
                    .dimensions_size() -
                batch_dims.size() !=
            2 ||
        (b_unary_ops.empty() ? b : b_unary_ops.back())
                    ->shape()
                    .dimensions_size() -
                batch_dims.size() !=
            2) {
      VLOG(1) << "Failed to rewrite " << instr->ToShortString()
              << "into FP8 Custom Call. A and B must have one non-contracting "
                 "dimension.";
      return false;
    }

    // Sequentially apply the collected unary and pad ops to the unconverted and
    // unscaled operands.
    auto shift_unary_ops =
        [&instr](HloInstruction *&x,
                 std::vector<HloInstruction *> &x_unary_ops) -> void {
      for (HloInstruction *unary_op : x_unary_ops) {
        std::vector<HloInstruction *> operands = {x};
        if (unary_op->opcode() == HloOpcode::kPad) {
          HloInstruction *convert =
              instr->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(unary_op->operand(1)->shape(),
                                               x->shape().element_type()),
                  unary_op->mutable_operand(1)));
          operands.emplace_back(convert);
        }
        x = instr->AddInstruction(unary_op->CloneWithNewOperands(
            ShapeUtil::MakeShapeWithDenseLayout(
                x->shape().element_type(), unary_op->shape().dimensions(),
                unary_op->shape().layout().minor_to_major()),
            operands));
      }
      return;
    };
    shift_unary_ops(a, a_unary_ops);
    shift_unary_ops(b, b_unary_ops);

    TF_ASSIGN_OR_RETURN(bool a_is_col_major,
                        MatrixIsColumnMajor(*instr, gemm_backend_config, "a"));
    TF_ASSIGN_OR_RETURN(bool b_is_col_major,
                        MatrixIsColumnMajor(*instr, gemm_backend_config, "b"));

    DotDimensionNumbers *dim_nums =
        gemm_backend_config.mutable_dot_dimension_numbers();
    int batch_dim_offset = batch_dims.size();

    // cuBLASLt FP8 GEMM kernels currently require the first operand, i.e. A, to
    // be row-major. If A is column-major, swap the contracting and
    // non-contracting dimension and transpose the matrix to effectively make it
    // column-major.
    // TODO(philipphack): Remove once cuBLASLt supports A being column-major
    if (a_is_col_major) {
      CHECK(a_contracting_dims[0] == batch_dim_offset ||
            a_contracting_dims[0] == batch_dim_offset + 1);
      if (a_contracting_dims[0] == batch_dim_offset) {
        dim_nums->set_lhs_contracting_dimensions(0, batch_dim_offset + 1);
      } else {
        dim_nums->set_lhs_contracting_dimensions(0, batch_dim_offset);
      }
      a = TransposeMatrix(a, a_contracting_dims[0], batch_dims);
    }

    // Similarly, cuBLASLt requires the second operand to be column-major, so
    // make it column-major if it is currently row-major.
    if (!b_is_col_major) {
      CHECK(b_contracting_dims[0] == batch_dim_offset ||
            b_contracting_dims[0] == batch_dim_offset + 1);
      if (b_contracting_dims[0] == batch_dim_offset) {
        dim_nums->set_rhs_contracting_dimensions(0, batch_dim_offset + 1);
      } else {
        dim_nums->set_rhs_contracting_dimensions(0, batch_dim_offset);
      }
      b = TransposeMatrix(b, b_contracting_dims[0], batch_dims);
    }

    // Pad the non-batch dimensions of the operands to multiples of 16 as
    // required by cuBLASLt.
    auto pad_operand = [&instr, &batch_dims](HloInstruction *&x) -> void {
      PaddingConfig padding_config;
      Shape padded_shape = x->shape();
      for (int i = 0; i < x->shape().rank(); ++i) {
        auto dimension = padding_config.add_dimensions();
        if (!absl::c_linear_search(batch_dims, i)) {
          int64_t padded_dimension =
              RoundUpTo<int64_t>(x->shape().dimensions(i), 16);
          dimension->set_edge_padding_low(0);
          dimension->set_edge_padding_high(padded_dimension -
                                           x->shape().dimensions(i));
          dimension->set_interior_padding(0);
          padded_shape.set_dimensions(i, padded_dimension);
        }
      }
      if (!ShapeUtil::Equal(padded_shape, x->shape())) {
        HloInstruction *zero =
            instr->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(x->shape().element_type())));
        x = instr->AddInstruction(
            HloInstruction::CreatePad(padded_shape, x, zero, padding_config));
      }
      return;
    };

    // Get the possible padded shape.
    auto pad_shape = [&batch_dims](const Shape old_shape) {
      Shape padded_shape = old_shape;
      for (int i = 0; i < old_shape.rank(); ++i) {
        if (!absl::c_linear_search(batch_dims, i)) {
          int64_t padded_dimension =
              RoundUpTo<int64_t>(old_shape.dimensions(i), 16);
          padded_shape.set_dimensions(i, padded_dimension);
        }
      }
      return padded_shape;
    };

    pad_operand(a);
    pad_operand(b);
    Shape new_output_shape;
    if (c == nullptr) {
      new_output_shape = pad_shape(instr->shape());
    } else {
      pad_operand(c);
      new_output_shape = c->shape();
    }

    std::vector<HloInstruction *> operands_list = {
        a, b, scales_f32[0], scales_f32[1], one, one};
    if (c != nullptr) {
      operands_list.insert(operands_list.begin() + 2, c);
    }

    HloInstruction *new_custom_call =
        instr->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeShapeWithDenseLayout(
                instr->shape().element_type(), new_output_shape.dimensions(),
                instr->shape().layout().minor_to_major()), operands_list,
            kCublasLtMatmulF8CallTarget));

    TF_RETURN_IF_ERROR(
        new_custom_call->set_backend_config(gemm_backend_config));
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), new_custom_call));

    // Slice the result of the GEMM if the operands were padded.
    HloInstruction *slice = nullptr;
    if (new_output_shape.dimensions() != instr->shape().dimensions()) {
      std::vector<int64_t> start_indices(instr->shape().rank(), 0);
      std::vector<int64_t> strides(instr->shape().rank(), 1);
      slice = instr->AddInstruction(HloInstruction::CreateSlice(
          instr->shape(), new_custom_call, start_indices,
          instr->shape().dimensions(), strides));
    }
    TF_RETURN_IF_ERROR(
        ReplaceInstruction(instr, slice ? slice : new_custom_call));
    return true;
  }

  Status F8ConvertD(HloInstruction *instr, HloInstruction *existing_gemm,
                    HloInstruction *d_scale, HloInstruction *clamp_lower,
                    HloInstruction *clamp_upper, bool mult_scale = false) {
    // Verify the data types and the operands of clamp.
    if (instr->shape().element_type() == F8E4M3FN) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::max()))) {
        return OkStatus();
      }
    } else if (instr->shape().element_type() == F8E5M2) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::max()))) {
        return OkStatus();
      }
    } else {
      return OkStatus();
    }

    if (!ShapeUtil::IsScalar(d_scale->shape())) {
      return OkStatus();
    }

    // The possible second user of the GEMM must be the calculation of the
    // maximum of the absolute value of the result of the GEMM. Since it is
    // unknown in what form this operation will be used, it is identified in a
    // top-down approach by inspecting the users of the GEMM.
    const std::vector<HloInstruction *> gemm_users = existing_gemm->users();
    HloInstruction *reduce_damax = nullptr;
    if (gemm_users.size() == 2) {
      // In the presence of a ReLU activation, the abs instruction is elided
      // since abs(ReLU(x)) = ReLU(x).
      TF_ASSIGN_OR_RETURN(auto config,
                          existing_gemm->backend_config<GemmBackendConfig>());
      for (int i = 0; i < gemm_users.size(); ++i) {
        HloInstruction *maybe_reduce = nullptr;
        if (gemm_users[i]->opcode() == HloOpcode::kAbs) {
          if (gemm_users[i]->users().size() != 1) continue;
          maybe_reduce = gemm_users[i]->users()[0];
        } else {
          // If there is no Abs instruction, relu is required as epilogue to
          // ensure all values are nonnegative.
          if (config.epilogue() != GemmBackendConfig::BIAS_RELU &&
              config.epilogue() != GemmBackendConfig::RELU)
            continue;
          maybe_reduce = gemm_users[i];
        }

        if (maybe_reduce->opcode() == HloOpcode::kReduce &&
            maybe_reduce->operands().size() == 2 &&
            maybe_reduce->operand(1)->opcode() == HloOpcode::kConstant &&
            ShapeUtil::IsScalar(maybe_reduce->operand(1)->shape())) {
          HloInstruction *reduce = maybe_reduce;
          HloComputation *reduce_comp = reduce->to_apply();
          HloInstruction *reduce_comp_root = reduce_comp->root_instruction();
          if (reduce->operand(1)->literal().GetAsDouble({}) <= 0. &&
              reduce_comp_root->opcode() == HloOpcode::kMaximum &&
              reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
              reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter) {
            reduce_damax = reduce;
          }
        }
      }
      if (!reduce_damax) {
        return OkStatus();
      }
    } else if (gemm_users.size() > 2) {
      return OkStatus();
    }

    TF_ASSIGN_OR_RETURN(auto gemm_backend_config,
                        existing_gemm->backend_config<GemmBackendConfig>());
    if (gemm_backend_config.beta() == 1.0 &&
        existing_gemm->operand(2)->shape().element_type() != BF16 &&
        existing_gemm->operand(2)->shape().element_type() != F16) {
      VLOG(1) << "The scaling and conversion of the result of "
              << existing_gemm->ToShortString()
              << " is not fused into the FP8 Custom Call because it "
                 "conflicts with the existing fusion of the addition of a "
                 "matrix bias with element type other than BF16 or F16.";
      return OkStatus();
    }

    // If necessary, invert the scaling factor of D and convert to F32.
    if (!mult_scale) {
      Literal one_literal = LiteralUtil::One(d_scale->shape().element_type());
      HloInstruction *one = instr->AddInstruction(
          HloInstruction::CreateConstant(one_literal.Clone()));
      d_scale = instr->AddInstruction(HloInstruction::CreateBinary(
          d_scale->shape(), HloOpcode::kDivide, one, d_scale));
    }
    if (d_scale->shape().element_type() != F32) {
      d_scale = instr->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeScalarShape(F32), d_scale));
    }

    TF_RETURN_IF_ERROR(existing_gemm->ReplaceOperandWith(
        gemm_backend_config.beta() == 0.0 ? 5 : 6, d_scale));

    // If present, elide the calculation of the maximum of the absolute values
    // of the result of the GEMM.
    if (reduce_damax) {
      return F8AddDAmax(instr, existing_gemm, reduce_damax);
    }

    std::unique_ptr<HloInstruction> new_gemm =
        existing_gemm->CloneWithNewShape(instr->shape());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(new_gemm)));

    return OkStatus();
  }

  // Adds a scalar DAmax return value to an FP8 GEMM.
  Status F8AddDAmax(HloInstruction *instr, HloInstruction *existing_gemm,
                    HloInstruction *reduce_damax) {
    // Change the output shape of the Custom Call to tuple(D, DAmax).
    Shape damax_shape = ShapeUtil::MakeScalarShape(F32);
    Shape tuple_shape =
        ShapeUtil::MakeTupleShape({instr->shape(), damax_shape});
    HloInstruction *gemm_and_damax =
        instr->AddInstruction(existing_gemm->CloneWithNewShape(tuple_shape));

    // Obtain D and DAmax separately from the output tuple.
    HloInstruction *d =
        instr->AddInstruction(HloInstruction::CreateGetTupleElement(
            instr->shape(), gemm_and_damax, 0));
    HloInstruction *damax = instr->AddInstruction(
        HloInstruction::CreateGetTupleElement(damax_shape, gemm_and_damax, 1));

    // Convert DAmax from FP32 to the requested type and elide reduce.
    HloInstruction *damax_converted = instr->AddInstruction(
        HloInstruction::CreateConvert(reduce_damax->shape(), damax));
    TF_RETURN_IF_ERROR(ReplaceInstruction(reduce_damax, damax_converted));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, d));

    return OkStatus();
  }

  Status FuseMatrixBiasAdd(HloInstruction *instr, HloInstruction *bias,
                           const HloInstruction *gemm,
                           HloInstruction *bitcast = nullptr) {
    TF_RET_CHECK(bias->shape() == (bitcast ? bitcast->shape() : gemm->shape()));

    // Do not fuse bias into S32 GEMM, as for this datatype cuBLAS only
    // supports fixed values for alpha/beta.
    if (gemm->shape().element_type() == S32) {
      return OkStatus();
    }

    // Cublas gemm overwrites the bias matrix, so fusion is only possible if the
    // gemm is the only user. CublasLt gemm can operate out-of-place.
    bool can_overwrite_bias = [bias]() {
      if (bias->user_count() > 1) {
        // There is another user of the data, do not overwrite it.
        return false;
      }

      if (bias->opcode() != HloOpcode::kParameter) {
        // Not a parameter; can overwrite.
        return true;
      }

      // The bias is a parameter of the computation; check if it is aliased.
      if (!bias->parent()->IsEntryComputation()) {
        // Only the HloModule has input/output aliasing, since this is not the
        // entry computation, there are no guarantees about aliasing; do not
        // overwrite.
        return false;
      }
      const auto &in_out_alias_config =
          bias->GetModule()->input_output_alias_config();
      // If the parameter is aliased, we can overwrite it.
      // TODO(victorstone): The assumption when calling ParameterHasAlias is
      // that bias is not a tuple. This is why we pass {} as the argument for
      // param_index.
      return in_out_alias_config.ParameterHasAlias(bias->parameter_number(),
                                                   /*param_index=*/{});
    }();
    bool want_to_fuse_bias = IsCublasLtMatmul(*gemm) || can_overwrite_bias;

    auto config = gemm->backend_config<GemmBackendConfig>().value();

    // It is possible to fuse into a cublasLt matmul that already has a vector
    // bias, but no other epilogue will commute with the matrix bias add.
    bool supported_epilogue =
        ((config.epilogue() == GemmBackendConfig::DEFAULT) ||
         (config.epilogue() == GemmBackendConfig::BIAS));

    if ((config.beta() != 0) || !want_to_fuse_bias ||
        (gemm->user_count() != 1) || !supported_epilogue) {
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
    if (IsLegacyCublasMatmul(*fused_op) || can_overwrite_bias) {
      xla::Cast<HloCustomCallInstruction>(fused_op.get())
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));

    if (bitcast != nullptr) {
      fused_op = bitcast->CloneWithNewOperands(
          bitcast->shape(),
          {bitcast->parent()->AddInstruction(std::move(fused_op))});
    }

    return ReplaceWithNewInstruction(instr, std::move(fused_op));
  }

  StatusOr<bool> FuseVectorBiasAdd(HloInstruction *instr,
                                   HloInstruction *broadcast,
                                   HloInstruction *gemm,
                                   HloInstruction *slice = nullptr,
                                   HloInstruction *convert = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast->shape(), (slice ? slice->shape() : gemm->shape())));

    // Verify that the data type is supported by Epilogue Fusion.
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return false;
    }

    HloInstruction *bias = broadcast->mutable_operand(0);

    TF_ASSIGN_OR_RETURN(auto config, gemm->backend_config<GemmBackendConfig>());

    // # output column dims == # non-contracting rhs operand dims.
    const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
    size_t num_col_dims = gemm->operand(1)->shape().rank() -
                          dot_dims.rhs_batch_dimensions_size() -
                          dot_dims.rhs_contracting_dimensions_size();

    if ((gemm->user_count() != 1) ||
        (config.epilogue() != GemmBackendConfig::DEFAULT) ||
        (bias->shape().rank() != num_col_dims)) {
      return false;
    }
    // We require the bias vector to have been broadcast in the most major
    // dimensions; i.e. its most minor physical dimensions align with most minor
    // physical dimensions of the gemm output.
    absl::Span<const int64_t> broadcast_dims = broadcast->dimensions();
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

    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    // When (non-trivial) matrix and vector bias co-exist for FP8 matmul, just
    // fuse matrix bias.
    if (gemm->custom_call_target() == kCublasLtMatmulF8CallTarget &&
        config.beta() != 0.0) {
      return true;
    }

    if (gemm->custom_call_target() == kCublasLtMatmulF8CallTarget &&
        bias->shape().element_type() == F32) {
      if (convert == nullptr) {
        return false;
      }

      HloInstruction *bias_f16_or_bf16 = convert->mutable_operand(0);
      auto compatible_bias_type = [](const PrimitiveType bias_type,
                                     const PrimitiveType output_type) {
        if (bias_type == BF16) {
          return output_type == F8E4M3FN || output_type == F8E5M2 ||
                 output_type == F32 || output_type == BF16;
        } else if (bias_type == F16) {
          return output_type == F16 || output_type == F8E4M3FN ||
                 output_type == F8E5M2;
        }
        return false;
      };

      // cuBLAS LT does not support FP32 biases on matmuls with FP8 inputs,
      // even if the matmul output is FP32. We do not unconditionally convert
      // the bias to a supported precision (F16 or BF16) because this lowers
      // precision. Instead, we only fuse the bias if the bias itself is a
      // convert from F16 or BF16, fusing the input of the convert instruction
      // to the matmul.
      if (compatible_bias_type(bias_f16_or_bf16->shape().element_type(),
                               gemm->shape().element_type())) {
        bias = bias_f16_or_bf16;
      } else {
        VLOG(1) << "Epilogue fusion of FP32 vector bias into FP8 GEMM is "
                   "currently not supported. See the cublasLT support matrix.";
        return false;
      }
    }

    // Replace add(gemm, broadcast) with fused new_gemm.
    operands.push_back(bias);
    config.set_epilogue(GemmBackendConfig::BIAS);
    std::unique_ptr<HloInstruction> result =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    TF_RETURN_IF_ERROR(result->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));
    if (slice != nullptr) {
      result = slice->CloneWithNewOperands(
          slice->shape(), {slice->parent()->AddInstruction(std::move(result))});
    }

    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(result)));
    return true;
  }

  Status FuseReluActivation(HloInstruction *instr, HloInstruction *broadcast,
                            HloInstruction *gemm,
                            HloInstruction *slice_or_bitcast = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast->shape(),
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

    std::unique_ptr<HloInstruction> result = gemm->Clone();
    TF_RETURN_IF_ERROR(result->set_backend_config(config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));

    if (slice_or_bitcast != nullptr) {
      result = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {slice_or_bitcast->parent()->AddInstruction(std::move(result))});
    }

    return ReplaceWithNewInstruction(instr, std::move(result));
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
  GpuVersion gpu_version_;

  // Choose cublas or cublasLt for the target of the custom call that instr will
  // be rewritten into.
  StatusOr<absl::string_view> GetNonFp8GemmCustomCallTarget(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    if (!instr.GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_cublaslt()) {
      // cublasLt is not enabled.
      return absl::string_view(kGemmCallTarget);
    }

    // cublasLt is enabled, check if other internal conditions are met.
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);
    if (lhs->shape().element_type() == S8 ||
        rhs->shape().element_type() == S8) {
      // TODO(b/241446501) The XLA usage of cublasLt does not yet handle
      // int8 matmuls. Fallback to legacy cublas.
      return absl::string_view(kGemmCallTarget);
    }

    // All internal conditions are met, check if we meet the requirements of
    // cublasLt.
    TF_ASSIGN_OR_RETURN(bool gemm_is_supported_by_cublas_lt,
                        GemmIsSupportedByCublasLt(instr, gemm_backend_config));
    if (gemm_is_supported_by_cublas_lt) {
      return absl::string_view(kCublasLtMatmulCallTarget);
    }

    // This case is not supported by cublasLt, fallback to legacy cublas.
    return absl::string_view(kGemmCallTarget);
  }

  StatusOr<bool> TypesAreSupportedByCublasLt(
      const HloInstruction &instr) const {
    // Figure out the Atype/Btype.
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    // cublasLt has a defined set of combinations of types that it supports.
    // Figure out the computeType and scaleType.
    TF_ASSIGN_OR_RETURN(const se::blas::DataType output_dtype,
                        AsBlasDataType(instr.shape().element_type()));
    TF_ASSIGN_OR_RETURN(const se::blas::ComputationType compute_type,
                        GetBlasComputationType(
                            a_dtype, instr.shape().element_type(),
                            stream_executor::blas::kDefaultComputePrecision));
    se::blas::DataType scale_type =
        cublas_lt::GetScaleType(output_dtype, compute_type);

    using se::blas::ComputationType;
    using se::blas::DataType;
    // This matrix of supported types is taken directly from cublasLt
    // documentation.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    const std::array<
        std::tuple<ComputationType, DataType /*scale_type*/,
                   PrimitiveType /*a_dtype*/, PrimitiveType /*b_dtype*/,
                   DataType /*output_dtype*/>,
        32>
        supported_type_combinations = {{
            // FP8 types:
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E4M3FN, DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E4M3FN, DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E4M3FN, DataType::kFloat},

            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E5M2, DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E5M2, DataType::kF8E4M3FN},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E5M2, DataType::kF8E5M2},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E5M2, DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
             PrimitiveType::F8E5M2, DataType::kFloat},

            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
             PrimitiveType::F8E4M3FN, DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
             PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
             PrimitiveType::F8E4M3FN, DataType::kF8E5M2},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
             PrimitiveType::F8E4M3FN, DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
             PrimitiveType::F8E4M3FN, DataType::kFloat},

            // Other data types:
            {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kHalf},

            {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kInt32},
            {ComputationType::kI32, DataType::kFloat, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kInt8},

            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             PrimitiveType::BF16, DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             PrimitiveType::BF16, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},

            // There would be an entry here for A/BType complex int8, but we do
            // not support that type.
            {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
             PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kBF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kTF32AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},

            {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
             PrimitiveType::F64, DataType::kDouble},
            {ComputationType::kF64, DataType::kComplexDouble,
             PrimitiveType::C128, PrimitiveType::C128,
             DataType::kComplexDouble},
        }};

    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
  }

  StatusOr<bool> MatrixIsColumnMajor(
      const HloInstruction &instr, const GemmBackendConfig &gemm_backend_config,
      const std::string matrix_name = "output") const {
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);

    const DotDimensionNumbers &dot_dims =
        gemm_backend_config.dot_dimension_numbers();
    TF_ASSIGN_OR_RETURN(
        GemmConfig gemm_config,
        GemmConfig::For(
            lhs->shape(), dot_dims.lhs_batch_dimensions(),
            dot_dims.lhs_contracting_dimensions(), rhs->shape(),
            dot_dims.rhs_batch_dimensions(),
            dot_dims.rhs_contracting_dimensions(),
            /*output_shape=*/instr.shape(), gemm_backend_config.alpha_real(),
            gemm_backend_config.alpha_imag(), gemm_backend_config.beta(),
            /*algorithm*/ std::nullopt, se::blas::kDefaultComputePrecision));

    if (matrix_name == "lhs" || matrix_name == "a") {
      return gemm_config.lhs_layout.order == MatrixLayout::Order::kColumnMajor;
    } else if (matrix_name == "rhs" || matrix_name == "b") {
      return gemm_config.rhs_layout.order == MatrixLayout::Order::kColumnMajor;
    } else if (matrix_name == "output" || matrix_name == "d") {
      return gemm_config.output_layout.order ==
             MatrixLayout::Order::kColumnMajor;
    } else {
      return InternalError("Invalid matrix name.");
    }
  }

  StatusOr<bool> GemmIsSupportedByCublasLt(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);
    const Shape &output_shape = instr.shape();

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

    auto cuda_compute_capability_ =
        std::get<se::CudaComputeCapability>(gpu_version_);
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

    TF_ASSIGN_OR_RETURN(bool output_is_column_major,
                        MatrixIsColumnMajor(instr, gemm_backend_config));
    if (!output_is_column_major) {
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

  // Turns an F8 dot into an F16 dot, converting operands to F16 and
  // converting the output back to F8.
  StatusOr<HloInstruction *> TurnF8DotIntoF16Dot(HloInstruction *instr) {
    DCHECK(IsF8Type(instr));
    DCHECK(IsF8Type(instr->operand(0)));
    DCHECK(IsF8Type(instr->operand(1)));

    // Convert operands to F16
    for (int i = 0; i < 2; ++i) {
      Shape operand_f16_shape = instr->operand(i)->shape();
      operand_f16_shape.set_element_type(F16);
      HloInstruction *convert =
          instr->AddInstruction(HloInstruction::CreateConvert(
              operand_f16_shape, instr->mutable_operand(i)));
      TF_RETURN_IF_ERROR(instr->ReplaceOperandWith(i, convert));
    }

    // Clone instruction and convert output to F8
    Shape output_f16_shape = instr->shape();
    output_f16_shape.set_element_type(F16);
    HloInstruction *f16_dot =
        instr->AddInstruction(instr->CloneWithNewShape(output_f16_shape));
    HloInstruction *convert_to_f8 = instr->AddInstruction(
        HloInstruction::CreateConvert(instr->shape(), f16_dot));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, convert_to_f8));
    return f16_dot;
  }
};

StatusOr<bool> RunOnComputation(HloComputation *computation,
                                GpuVersion gpu_version) {
  GemmRewriterVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

GemmRewriter::GemmRewriter(GpuVersion gpu_version)
    : gpu_version_(gpu_version) {}

StatusOr<bool> GemmRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
