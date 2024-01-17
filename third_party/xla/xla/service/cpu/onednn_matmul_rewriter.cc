/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_matmul_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace cpu {

namespace {
namespace m = match;

inline Status ValidateDotDimensionNumbers(
    const DotDimensionNumbers& dim_numbers) {
  // Checks some invariants that do not hold in general, but DotDecomposer
  // should have established for us.
  TF_RET_CHECK(dim_numbers.lhs_contracting_dimensions_size() == 1);
  std::vector<int64_t> batch_dim_numbers(
      dim_numbers.lhs_batch_dimensions_size());
  absl::c_iota(batch_dim_numbers, 0);
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.lhs_batch_dimensions()));
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.rhs_batch_dimensions()));
  return OkStatus();
}

template <typename Pattern>
inline auto AllowedIntermediateInstructions(HloInstruction** bitcast,
                                            Pattern pattern) {
  // Checks the presence of some intermediate operations that can be moved /
  // folded to allow dot fusion with add.
  // We try to match either of the following:
  // 1. custom-call -> bf16-to-fp32 convert
  // 2. custom-call -> bf16-to-fp32 convert -> bitcast
  auto common =
      m::AnyOf<HloInstruction>(m::Convert(std::move(pattern).WithOneUser())
                                   .WithElementType(PrimitiveType::F32),
                               std::move(pattern).WithOneUser())
          .WithOneUser();
  return m::AnyOf<HloInstruction>(m::Bitcast(bitcast, common), common)
      .WithOneUser();
}

// We also check if the convert instruction has only one use.
inline bool AllOperandsConvertedFromBF16ToF32(const HloInstruction* instr) {
  return absl::c_all_of(instr->operands(), [](HloInstruction* operand) {
    return Match(operand,
                 m::Convert(m::Op().WithElementType(PrimitiveType::BF16))
                     .WithElementType(PrimitiveType::F32)
                     .WithOneUse());
  });
}

template <typename Pattern>
auto ElementwiseSafeIntermediate(HloInstruction** instr, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Broadcast(instr, pattern.WithOneUser()),
                                  m::Slice(instr, pattern.WithOneUser()),
                                  m::Bitcast(instr, pattern.WithOneUser()),
                                  m::Reshape(instr, pattern.WithOneUser()),
                                  pattern);
}

inline auto OneDnnMatmulInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$matmul"});
}

inline auto ConvertBF16ToF32(HloInstruction** instr) {
  return m::Convert(m::Op(instr).WithElementType(PrimitiveType::BF16))
      .WithElementType(PrimitiveType::F32);
}

inline void GetBF16Bias(HloInstruction* dot, HloInstruction** old_bias,
                        HloInstruction** new_bias) {
  if (dot->shape().element_type() == PrimitiveType::BF16 &&
      (((*old_bias)->operand_count() == 1 &&
        Match((*old_bias)->mutable_operand(0), ConvertBF16ToF32(new_bias))) ||
       Match(*old_bias, ConvertBF16ToF32(new_bias)))) {
    *old_bias = *new_bias;
  }
}

inline auto BcastConstScalar(HloInstruction** instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

inline auto BcastConstScalar(double value) {
  return BcastConstScalar(nullptr, value);
}

auto ConstScalarNear(double value) {
  return m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction* instr) {
        // Not a very robust floating-point comparison, but good enough for our
        // purposes.
        std::optional<double> actual =
            static_cast<const HloConstantInstruction*>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) return false;
        double epsilon;
        switch (instr->shape().element_type()) {
          case F16:
            epsilon = 128 * std::numeric_limits<Eigen::half>::epsilon();
            break;
          case BF16:
            epsilon = 128 * std::numeric_limits<bfloat16>::epsilon();
            break;
          case F32:
            epsilon = 128 * std::numeric_limits<float>::epsilon();
            break;
          case F64:
            epsilon = 128 * std::numeric_limits<double>::epsilon();
            break;
          default:
            return false;
        }
        return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
      });
}

inline auto BcastConstScalarNear(double value) {
  return m::Broadcast(ConstScalarNear(value));
}

auto GELUActivation(HloInstruction* instr, HloInstruction** src) {
  // Attempt to match GELU_TANH activation
  // (https://arxiv.org/abs/1606.08415), where:
  // gelu_tanh(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
  HloInstruction* errf;
  return Match(instr, m::MultiplyAnyOrder(
                          m::Op(src),
                          m::MultiplyAnyOrder(
                              BcastConstScalar(0.5),
                              m::AddAnyOrder(BcastConstScalar(1.0),
                                             m::Op(&errf).WithOneUser())))) &&
         Match(errf,
               m::Tanh(m::MultiplyAnyOrder(
                           BcastConstScalarNear(sqrt(M_2_PI)),
                           m::AddAnyOrder(
                               m::Op().Is(*src),
                               m::MultiplyAnyOrder(
                                   BcastConstScalarNear(0.044715),
                                   m::MultiplyAnyOrder(
                                       m::Op().Is(*src),
                                       m::MultiplyAnyOrder(m::Op().Is(*src),
                                                           m::Op().Is(*src))
                                           .WithOneUser())
                                       .WithOneUser())
                                   .WithOneUser())
                               .WithOneUser())
                           .WithOneUser())
                   .WithOneUser());
}

StatusOr<Shape> AdjustBiasShape(const HloInstruction* instr,
                                const absl::Span<const int64_t>& dot_dims) {
  auto bcast = Cast<HloBroadcastInstruction>(instr);
  // OneDNN matmul apply auto-broadcast to BIAS
  // Replace Broadcast with Bitcast to match rank
  Shape new_shape = bcast->shape();
  auto kept_dimensions = bcast->dimensions();
  for (int i = 0; i < new_shape.rank(); i++) {
    if (!absl::c_linear_search(kept_dimensions, i)) {
      new_shape.set_dimensions(i, 1);
    }
  }

  // Remove dimensions with value=1 to match dot dimensions
  bool more_dims_to_delete = true;
  while (more_dims_to_delete && new_shape.rank() > dot_dims.size()) {
    for (int64_t i = 0; i < new_shape.rank(); i++) {
      if (new_shape.dimensions()[i] == 1) {
        new_shape.DeleteDimension(i);
        break;
      }
    }
    more_dims_to_delete = false;
  }

  // Validate dimensions
  auto new_shape_dims = new_shape.dimensions();
  for (int i = 0; i < dot_dims.size() - 2; i++) {
    if (new_shape_dims[i] != 1 && new_shape_dims[i] != dot_dims[i]) {
      return absl::CancelledError("bias cannot be fused - dimensions mismatch");
    }
  }
  return new_shape;
};

}  // namespace

bool OneDnnMatMulRewriter::ShouldRewrite(const HloInstruction* dot_instr) {
  // Currently, blocking control dependencies
  if (dot_instr->HasControlDependencies()) return false;
  if (!IsSupportedType(dot_instr->shape().element_type())) return false;
  if (dot_instr->operands().size() != 2) return false;

  // Currently, we rewrite when the data type is F32 or BF16. Note we do not
  // need to check equality of contraction dim-size of the operands. HLO
  // verifier already does the job. We, however, need to check if contraction
  // is over only 1 dimension (a.k.a. K dimension in matrix-multiplication
  // parlance). We also restrict that batch dimensions of the operands
  // match.
  const Shape& lhs_shape = dot_instr->operand(0)->shape();
  const Shape& rhs_shape = dot_instr->operand(1)->shape();
  const Shape& output_shape = dot_instr->shape();
  // None of the operands and result should be ZeroElementArray.
  if (ShapeUtil::IsZeroElementArray(lhs_shape) ||
      ShapeUtil::IsZeroElementArray(rhs_shape) ||
      ShapeUtil::IsZeroElementArray(output_shape)) {
    return false;
  }
  // OneDNN only supports 2 <= rank <= kOneDnnMaxNDims.
  if (lhs_shape.rank() != rhs_shape.rank() ||
      rhs_shape.rank() != output_shape.rank() || lhs_shape.rank() < 2 ||
      lhs_shape.rank() > kOneDnnMaxNDims) {
    return false;
  }
  // Layout should be row-major, contraction dimensions captures transpose
  // scenarios in last two dimensions.
  if (!LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout())) {
    return false;
  }

  auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
  int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);
  // Supported contraction is only in one of last two dimensions.
  if (lhs_dim_k < lhs_shape.rank() - 2 || rhs_dim_k < rhs_shape.rank() - 2) {
    return false;
  }

  // OneDNN matmul has scratch allocation and copy overheads. The overheads
  // can be amortized if there is sufficient MAC (multiply-accumulate)
  // operations. We don't rewrite for small cases (determined empirically).
  // TODO(intel-tf): Relax the condition when more optimizations in oneDNN
  // matmul is achieved.
  auto rank = lhs_shape.rank();
  auto rhs_dims = rhs_shape.dimensions();
  int64_t num_mac_ops = ShapeUtil::ElementsIn(lhs_shape) * rhs_dims.back();
  int mac_ops_threshold = (rank == 2) ? (1 << 23) : (1 << 18);
  return (num_mac_ops >= mac_ops_threshold);
}

class OneDnnMatMulRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  // Matches patterns for possible MatMul fusions that are supported by oneDNN
  // library. Matched HLO instruction(s) are replaced by custom call.
  Status HandleDot(HloInstruction* instr) override {
    HloInstruction* dot_instr;
    auto pattern = m::Op(&dot_instr).WithOpcode(HloOpcode::kDot);
    if (!Match(instr, pattern)) return OkStatus();

    auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
    TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(dot_dim_numbers));
    if (!OneDnnMatMulRewriter::ShouldRewrite(dot_instr)) return OkStatus();
    const Shape& lhs_shape = dot_instr->operand(0)->shape();
    const Shape& rhs_shape = dot_instr->operand(1)->shape();
    const Shape& output_shape = dot_instr->shape();

    int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
    int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);

    HloInstruction* matmul_call =
        dot_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape,
            {dot_instr->mutable_operand(0), dot_instr->mutable_operand(1)},
            "__onednn$matmul"));
    // Set additional info via config, e.g., transpose and fusion info.
    BackendConfig backend_config;
    OneDnnMatMulConfig* matmul_config =
        backend_config.mutable_onednn_matmul_config();
    bool transpose_a = (lhs_dim_k != lhs_shape.rank() - 1);
    bool transpose_b = (rhs_dim_k != rhs_shape.rank() - 2);
    matmul_config->set_transpose_a(transpose_a);
    matmul_config->set_transpose_b(transpose_b);
    TF_RETURN_IF_ERROR(matmul_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, matmul_call));
    return OkStatus();
  }

  Status HandleConvert(HloInstruction* convert) override {
    HloInstruction* matmul_instr;
    auto pattern =
        m::Convert(m::CustomCall(&matmul_instr, {"__onednn$matmul"})
                       .WithOneUse()
                       .WithElementType(PrimitiveType::F32)
                       .WithPredicate(AllOperandsConvertedFromBF16ToF32))
            .WithElementType(PrimitiveType::BF16);

    if (!Match(convert, pattern)) return OkStatus();
    if (!IsSupportedType(convert->shape().element_type())) return OkStatus();

    // BFloat16 operands.
    std::vector<HloInstruction*> bf16_operands;
    for (auto operand : matmul_instr->operands()) {
      bf16_operands.push_back(operand->mutable_operand(0));
    }

    HloInstruction* matmul_call = convert->AddInstruction(
        matmul_instr->CloneWithNewOperands(convert->shape(), bf16_operands));
    TF_RETURN_IF_ERROR(ReplaceInstruction(convert, matmul_call));
    return OkStatus();
  }

  Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *addend, *dot, *bcast_input, *bitcast_input;
    HloInstruction* bitcast = nullptr;

    auto pattern =
        m::Op(&instr)
            .WithOpcode(HloOpcode::kAdd)
            .WithBinaryOperandsAnyOrder(
                AllowedIntermediateInstructions(
                    &bitcast, m::Op(&dot)
                                  .WithOneUser()
                                  .WithOpcode(HloOpcode::kCustomCall)
                                  .WithCustomCallTarget({"__onednn$matmul"})),
                m::Op(&addend).WithOneUser());

    auto nonscalar_broadcast =
        m::Broadcast(m::Op(&bcast_input)
                         .WithPredicate([](const HloInstruction* ins) {
                           return !ShapeUtil::IsEffectiveScalar(ins->shape());
                         })
                         .WithOneUser())
            .WithOneUser();

    auto addend_reshape =
        m::Bitcast(m::Op(&bitcast_input).WithOneUser()).WithOneUser();

    if (Match(instr, pattern)) {
      if (!IsSupportedType(dot->shape().element_type())) return OkStatus();
      if (!dot->backend_config<BackendConfig>()
               ->mutable_onednn_matmul_config()
               ->fused_ops()
               .empty() &&
          dot->backend_config<BackendConfig>()
                  ->mutable_onednn_matmul_config()
                  ->fused_ops(0) == OneDnnMatMulConfig::BIAS) {
        return OkStatus();
      }
      std::vector<HloInstruction*> new_operands;
      for (auto operand : dot->operands()) {
        new_operands.push_back(operand);
      }

      // For addend, match one of these patterns:
      // 1. any(addend) -> broadcast -> add
      // 2. any(addend) -> bitcast -> add
      bool bias_bcast = Match(addend, nonscalar_broadcast);
      bool check_addend = Match(addend, addend_reshape);

      HloInstruction *bf16_addend, *bf16_bcast_input, *bf16_bitcast_input;
      // If addend is bf16 and being converted to f32, get the original bf16
      // one.
      GetBF16Bias(dot, &addend, &bf16_addend);
      if (bias_bcast) {
        GetBF16Bias(dot, &bcast_input, &bf16_bcast_input);
      }
      if (check_addend) {
        GetBF16Bias(dot, &bitcast_input, &bf16_bitcast_input);
      }

      if (bias_bcast) {
        if (bcast_input->shape().rank() == 1) {
          new_operands.push_back(bcast_input);
        } else {
          auto new_shape = AdjustBiasShape(addend, dot->shape().dimensions());
          if (new_shape.ok()) {
            auto bitcast = bcast_input->AddInstruction(
                HloInstruction::CreateBitcast(new_shape.value(), bcast_input));
            new_operands.push_back(bitcast);
          } else {
            LOG(WARNING) << new_shape.status();
            new_operands.push_back(nullptr);
          }
        }
      } else if (absl::c_equal(dot->shape().dimensions(),
                               addend->shape().dimensions())) {
        new_operands.push_back(addend);
      } else if (check_addend &&
                 absl::c_equal(dot->shape().dimensions(),
                               bitcast_input->shape().dimensions())) {
        new_operands.push_back(bitcast_input);
      } else {
        new_operands.push_back(nullptr);
      }

      if (new_operands.back() == nullptr) {
        return OkStatus();
      }

      bool nd_bias = absl::c_count_if(new_operands.back()->shape().dimensions(),
                                      [](int64_t dim) { return dim > 1; }) > 1;

      auto matmul_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), new_operands)));

      auto backend_config = matmul_call->backend_config<BackendConfig>();
      backend_config->mutable_onednn_matmul_config()->add_fused_ops(
          nd_bias ? OneDnnMatMulConfig::BINARY_ADD : OneDnnMatMulConfig::BIAS);
      backend_config->mutable_onednn_matmul_config()->set_bias_broadcast(
          bias_bcast);

      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));

      HloInstruction* new_instr;
      // If matched pattern has custom-call -> bitcast -> add, then we need to
      // insert bitcast after the new fusion to maintain the correct shape
      // (new-custom-call -> bitcast). Also, this will be followed by -> convert
      // for bf16 case to avoid datatype mismatch.
      if (bitcast != nullptr && bitcast->opcode() == HloOpcode::kBitcast) {
        if (matmul_call->shape().element_type() == PrimitiveType::BF16) {
          auto bitcast_call =
              matmul_call->AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::ChangeElementType(instr->shape(),
                                               PrimitiveType::BF16),
                  matmul_call));
          new_instr =
              bitcast_call->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(bitcast_call->shape(),
                                               PrimitiveType::F32),
                  bitcast_call));
        } else {
          new_instr = matmul_call->AddInstruction(
              HloInstruction::CreateBitcast(instr->shape(), matmul_call));
        }
      } else {
        if (matmul_call->shape().element_type() == PrimitiveType::BF16) {
          new_instr = matmul_call->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(matmul_call->shape(),
                                           PrimitiveType::F32),
              matmul_call));
        } else {
          new_instr = matmul_call;
        }
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }

    return OkStatus();
  }

  Status HandleMaximum(HloInstruction* instr) override {
    HloInstruction* matmul_call;
    HloInstruction* intermediate_instr = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM, including
    // when slicing or bitcasting is applied to the result.
    if (Match(instr, m::MaximumAnyOrder(ElementwiseSafeIntermediate(
                                            &intermediate_instr,
                                            OneDnnMatmulInstr(&matmul_call))
                                            .WithOneUser(),
                                        BcastConstScalar(0)))) {
      return FuseActivation(OneDnnMatMulConfig::RELU, instr, matmul_call,
                            intermediate_instr);
    }
    return OkStatus();
  }

  Status HandleMultiply(HloInstruction* instr) override {
    HloInstruction* matmul_call;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* src;
    if (GELUActivation(instr, &src)) {
      if (Match(src,
                ElementwiseSafeIntermediate(&intermediate_instr,
                                            OneDnnMatmulInstr(&matmul_call)))) {
        return FuseActivation(OneDnnMatMulConfig::GELU_TANH, instr, matmul_call,
                              intermediate_instr);
      }
    }
    return OkStatus();
  }

  Status FuseActivation(OneDnnMatMulConfig_FusionKind kind,
                        HloInstruction* activation, HloInstruction* matmul,
                        HloInstruction* intermediate_instr = nullptr) {
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        matmul->backend_config<BackendConfig>());
    auto* matmul_config = backend_config.mutable_onednn_matmul_config();
    matmul_config->add_fused_ops(kind);

    std::unique_ptr<HloInstruction> output = matmul->Clone();
    TF_RETURN_IF_ERROR(output->set_backend_config(backend_config));

    if (intermediate_instr) {
      output = intermediate_instr->CloneWithNewOperands(
          intermediate_instr->shape(),
          {matmul->parent()->AddInstruction(std::move(output))});
    }

    return ReplaceWithNewInstruction(activation, std::move(output));
  }
};

StatusOr<bool> OneDnnMatMulRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnMatMulRewriteVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
