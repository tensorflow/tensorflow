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

// OneDNN matmul can fuse add operation with automatic broadcasting along the
// addend's dimensions that are 1s. When compatible, Broadcast can be replaced
// by Bitcast, which is much cheaper. Compute new shape for the Bitcast.
StatusOr<Shape> AdjustBiasShape(const HloInstruction* broadcast_instr,
                                const Shape& dot_shape) {
  if (broadcast_instr->opcode() != HloOpcode::kBroadcast) {
    return absl::InvalidArgumentError(
        "Hlo instruction is not a Broadcast insruction.");
  }
  auto bcast = Cast<HloBroadcastInstruction>(broadcast_instr);
  Shape new_shape = bcast->shape();
  // Broadcast instruction has "dimensions" parameter along which its input's
  // dimensions should not change. For example,
  //      dot = f32[3,4,5,6] dot(...)
  //      arg = f32[3,6]{1,0} parameter(0)
  //      broad = f32[3,4,5,6]{3,2,1,0} broadcast(arg), dimensions={0,3}
  //      add = f32[3,4,5,6]{3,2,1,0} add(dot, arg)
  // can be replaced with the following
  //      arg = f32[3,6]{1,0} parameter(0)
  //      bitcast = f32[3,1,1,6]{3,2,1,0} bitcast(arg)
  //      fused = f32[3,4,5,6]{3,2,1,0} custom-call((..., bitcast)
  auto kept_dimensions = bcast->dimensions();
  for (int i = 0; i < new_shape.rank(); i++) {
    if (!absl::c_linear_search(kept_dimensions, i)) {
      new_shape.set_dimensions(i, 1);
    }
  }

  // If rank(new_shape) > rank(dot), extra dimensions with value = 1 can be
  // deleted from the new_shape.
  int64_t rank_difference = new_shape.rank() - dot_shape.rank();
  auto new_dims = new_shape.dimensions();
  std::vector<int64_t> dims_to_delete;
  for (int i = 0; i < rank_difference; ++i) {
    if (new_dims[i] == 1) {
      dims_to_delete.push_back(i);
    }
  }
  new_shape = ShapeUtil::DeleteDimensions(dims_to_delete, new_shape);

  // New shape for bias should satisfy the condition:
  //   rank(new_shape) <= rank(dot).
  if (new_shape.rank() > dot_shape.rank()) {
    return absl::CancelledError(
        "Bias shape could not be adjusted for a fusion.");
  }

  return new_shape;
};

inline bool IsOperandFusible(HloInstruction* operand, HloInstruction* dot) {
  // Check if the operand's shape is compatible with matmul for fusion.
  // An operand is fusable if
  //    1. rank(operand) <= rank(dot) and
  //    2. Starting from the last dim in backward direction, the dimension
  //       size of operand is either 1 or same to dot.
  auto operand_dims = operand->shape().dimensions();
  auto dot_dims = dot->shape().dimensions();
  if (operand_dims.size() > dot_dims.size()) return false;
  int operand_idx = operand_dims.size() - 1;
  int dot_idx = dot_dims.size() - 1;
  for (; operand_idx >= 0; --operand_idx, --dot_idx) {
    if (operand_dims[operand_idx] != 1 &&
        operand_dims[operand_idx] != dot_dims[dot_idx])
      return false;
  }
  return true;
}

inline bool IsRowMajor(const Shape& shape) {
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

// Whether the element type of instr is compatible with oneDNN kernels.
// TODO(intel-tf): Restict compatible types based on instruction kind.
inline bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F32;
}

// Type conversion from and to any of BF16 and FP32.
// TODO(intel-tf): Support more types when enabled.
template <typename Pattern>
inline auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

template <typename Pattern>
inline auto SupportedConvert(HloInstruction** convert, Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(convert, pattern).WithPredicate(supported_convert);
}

template <typename Pattern>
inline auto BitcastWithReshapeSemantics(HloInstruction** bitcast,
                                        Pattern pattern) {
  // TODO(intel-tf): Add stronger condition that Bitcast does not have transpose
  // semantics. Some of the HLO passes replaces Transpose with Bitcast. Here
  // the layouts are checked to be rowmajor since the current pass runs after
  // the layout assignment and oneDNN matmul is enabled for rowmajor layouts.
  auto is_reshape = [](const HloInstruction* instr) -> bool {
    if (!instr) return false;
    auto input_shape = instr->operand(0)->shape();
    auto output_shape = instr->shape();
    bool is_same_type = ShapeUtil::SameElementType(input_shape, output_shape);
    bool has_equal_num_elems = ShapeUtil::ElementsIn(input_shape) ==
                               ShapeUtil::ElementsIn(output_shape);
    bool has_rowmajor_layout =
        IsRowMajor(input_shape) && IsRowMajor(output_shape);
    return is_same_type && has_equal_num_elems && has_rowmajor_layout;
  };
  return m::Bitcast(bitcast, pattern).WithPredicate(is_reshape);
}

template <typename Pattern>
inline auto OptionalConvertAndBitcast(HloInstruction** optional_convert,
                                      HloInstruction** optional_bitcast,
                                      Pattern pattern) {
  // Checks the presence of some intermediate operations that can be moved /
  // folded to allow dot fusion with add.
  // Try to match either of the following:
  //   1. pattern-root -> bf16-to-fp32 convert -> bitcast
  //   2. pattern-root -> bf16-to-fp32 convert
  //   3. pattern-root -> bitcast
  //   4. pattern-root
  auto common =
      m::AnyOf<HloInstruction>(
          SupportedConvert(optional_convert, std::move(pattern).WithOneUser())
              .WithOperand(0, m::Op().WithElementType(PrimitiveType::BF16))
              .WithElementType(PrimitiveType::F32),
          std::move(pattern).WithOneUser())
          .WithOneUser();
  return m::AnyOf<HloInstruction>(
      BitcastWithReshapeSemantics(optional_bitcast, common), common);
}

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
  if (!IsRowMajor(lhs_shape) || !IsRowMajor(rhs_shape) ||
      !IsRowMajor(output_shape)) {
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
    // Try to do a fusion for Dot(onednn-matmul) + Add. However,
    // HLO Add instruction might receive the addends after additional
    // processing like Broadcast, Bitcast, Convert, etc. is applied to the raw
    // addends. Here, the following possible pattern is matched.
    //
    // clang-format off
    //
    //            Dot                               addend
    //             |                                  |
    //             v                                  v
    //     optional instructions            optional instructions
    //    (e.g, Convert, Bitcast)         (e.g, Convert, Broadcast)
    //             |                                  |
    //             +--------------+-------------------+
    //                            |
    //                            v
    //                           Add
    //
    // clang-format on

    HloInstruction *addend_intermediate, *dot;
    HloInstruction* optional_dot_bitcast = nullptr;
    HloInstruction* optional_dot_convert = nullptr;

    auto pattern = m::AddAnyOrder(
        &instr,
        OptionalConvertAndBitcast(&optional_dot_convert, &optional_dot_bitcast,
                                  OneDnnMatmulInstr(&dot))
            .WithOneUser(),
        m::Op(&addend_intermediate).WithOneUser());

    if (Match(instr, pattern)) {
      if (!IsSupportedType(dot->shape().element_type())) return OkStatus();
      // TODO(intel-tf): Remove the condition below when the fusion Dot +
      // Add(bias) + Add(e.g., residual) is enabled.
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

      // At this point, the addend could have one of the following
      // possiblities that the current fusion can handle:
      //
      //   - addend -> Convert -> Broadcast -> Add
      //   - addend -> Broadcast -> Convert -> Add
      //   - addend -> Convert
      //   - addend -> Broadcast
      //   - addend
      //
      // Hunt for addend through possible sequences above and check the addend
      // is compatible to onednn-matmul fusion.
      HloInstruction* addend = nullptr;
      HloInstruction* optional_addend_broadcast = nullptr;
      auto addend_pattern = m::AnyOf<HloInstruction>(
          m::Broadcast(&optional_addend_broadcast,
                       m::Convert(&addend, m::Op())),
          m::Convert(m::Broadcast(&optional_addend_broadcast, m::Op(&addend))),
          m::Convert(&addend, m::Op()),
          m::Broadcast(&optional_addend_broadcast, m::Op(&addend)),
          m::Op(&addend));
      if (!Match(addend_intermediate, addend_pattern)) return OkStatus();

      if (optional_addend_broadcast && addend->shape().rank() != 1) {
        auto new_shape =
            AdjustBiasShape(optional_addend_broadcast, dot->shape());
        if (new_shape.ok()) {
          addend = addend->AddInstruction(
              HloInstruction::CreateBitcast(new_shape.value(), addend));
        } else {
          VLOG(2) << new_shape.status();
          return OkStatus();
        }
      }

      // Validate addend for fusion.
      if (CompatibleElementType(addend) && IsOperandFusible(addend, dot)) {
        new_operands.push_back(addend);
      } else {
        return OkStatus();
      }

      auto matmul_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), new_operands)));

      auto backend_config = matmul_call->backend_config<BackendConfig>();
      backend_config->mutable_onednn_matmul_config()->add_fused_ops(
          addend->shape().rank() != 1 ? OneDnnMatMulConfig::BINARY_ADD
                                      : OneDnnMatMulConfig::BIAS);
      if (optional_addend_broadcast) {
        backend_config->mutable_onednn_matmul_config()->set_bias_broadcast(
            true);
      }
      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));

      HloInstruction* new_instr;
      // If matched pattern has custom-call -> bitcast -> add, then we need to
      // insert bitcast after the new fusion to maintain the correct shape
      // (new-custom-call -> bitcast). Also, this will be followed by -> convert
      // for bf16 case to avoid datatype mismatch.
      if (optional_dot_bitcast != nullptr &&
          optional_dot_bitcast->opcode() == HloOpcode::kBitcast) {
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
