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

#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#include "xla/service/cpu/onednn_contraction_rewriter.h"

#include "xla/executable_run_options.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_pattern_utils.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "xla/tsl/util/onednn_threadpool.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace cpu {

namespace {
namespace m = match;
namespace pu = ::xla::cpu::onednn_pattern_utils_internal;

inline absl::Status ValidateDotDimensionNumbers(
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
  return absl::OkStatus();
}

// Whether the element type of instr is compatible with oneDNN kernels.
// TODO(intel-tf): Restict compatible types based on instruction kind.
inline bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F32 || element_type == F16;
}

inline bool IsRowMajor(const Shape& shape) {
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
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
auto ElementwiseSafeIntermediates(HloInstruction** instr,
                                  HloInstruction** optional_bitcast,
                                  Pattern pattern) {
  return m::AnyOf<HloInstruction>(
      m::Broadcast(instr, pattern.WithOneUser()),
      m::Slice(instr, pattern.WithOneUser()),
      m::Bitcast(instr, pattern.WithOneUser()),
      m::Reshape(instr, pattern.WithOneUser()),
      pu::SupportedConvert(instr, pattern.WithOneUser()),
      pu::SupportedConvert(instr, BitcastWithReshapeSemantics(
                                      optional_bitcast, pattern.WithOneUser())),
      pattern);
}

inline auto OneDnnMatmulInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$matmul"});
}

inline auto OneDnnConvolutionInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$convolution"});
}

inline auto OneDnnFusibleInstr(HloInstruction** instr) {
  return m::AnyOf<HloInstruction>(
      m::CustomCall(instr, {"__onednn$matmul"}),
      m::CustomCall(instr, {"__onednn$convolution"}));
}

inline bool IsOneDnnMatmulInstr(const HloInstruction* instr) {
  return Match(instr, m::CustomCall({"__onednn$matmul"}));
}

inline bool IsOneDnnConvolutionInstr(const HloInstruction* instr) {
  return Match(instr, m::CustomCall({"__onednn$convolution"}));
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

inline auto BcastConvertConstScalar(double value) {
  return m::Broadcast(pu::OptionalConvert(m::ConstantScalar(value)));
}

inline bool IsBatchDot(const HloInstruction& instr) {
  if (auto* dot_instr = DynCast<HloDotInstruction>(&instr)) {
    return dot_instr->dot_dimension_numbers().lhs_batch_dimensions_size() > 0;
  }
  return false;
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

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsEffectiveScalar(instr->shape());
}

std::optional<float> GetConstantValueAsFloat32(const HloInstruction* inst) {
  if (!IsScalar(inst)) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return inst->literal().GetFirstElement<half>();
    case BF16:
      return inst->literal().GetFirstElement<bfloat16>();
    case F32:
      return inst->literal().GetFirstElement<float>();
    default:
      return std::nullopt;
  }
}

auto GetOneDnnContractionVariant(
    absl::StatusOr<BackendConfig>* backend_config) {
  return ((*backend_config)->backend_config_oneof_case() == kOnednnConvConfig)
             ? OneDnnContractionVariant(PrimitiveTrait<kOnednnConvConfig>{})
             : OneDnnContractionVariant(PrimitiveTrait<kOnednnMatmulConfig>{});
}

// Return the correct mutable config instance for the given contraction variant
// based on the template parameter
template <typename TransformationType>
TransformationType GetTransformationConfig(
    absl::StatusOr<BackendConfig>* backend_config) {
  return std::visit(
      [&](auto&& config) -> TransformationType {
        using T = std::decay_t<decltype(config)>;
        return PrimitiveTrait<T::kConfigVal, TransformationType>::
            GetTransformationConfig(
                GetKernelConfig<T::kConfigVal>(backend_config));
      },
      GetOneDnnContractionVariant(backend_config));
}

auto GetFusionsConfig(absl::StatusOr<BackendConfig>* backend_config) {
  return GetTransformationConfig<OneDnnFusionConfig*>(backend_config);
}

auto GetOptimizationsConfig(absl::StatusOr<BackendConfig>* backend_config) {
  return GetTransformationConfig<OneDnnOptimizationConfig*>(backend_config);
}

inline auto BcastConstScalarNear(double value) {
  return m::Broadcast(ConstScalarNear(value));
}

// Associativity and commutativity properties of multiply results in various
// patterns for an equivalent computation. This function tries to capture most
// of the variations for a computation a * b * c. For example, patterns could be
// any of (a * b) * c or a * (b * c), along with the variations resulting from
// commutative patterns.
template <typename PatternA, typename PatternB, typename PatternC>
inline auto MultiplyMultiplyAnyOrder(PatternA a, PatternB b, PatternC c) {
  return m::AnyOf<HloInstruction>(
      m::MultiplyAnyOrder(a, m::MultiplyAnyOrder(b, c)),
      m::MultiplyAnyOrder(b, m::MultiplyAnyOrder(a, c)),
      m::MultiplyAnyOrder(c, m::MultiplyAnyOrder(a, b)));
}

auto GELUActivation(HloInstruction* instr, HloInstruction** src) {
  // Attempt to match GELU_TANH activation or GELU_ERF activation
  // (https://arxiv.org/abs/1606.08415), where:
  // gelu_tanh(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
  //                     -------------errf_approximate------------
  //
  // gelu_erf(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
  //                     --errf_exact--

  HloInstruction* errf;

  // The expression 0.5 * x * (1 + errf) as common pattern for GELU exact and
  // approximate activations.
  auto common_pattern = MultiplyMultiplyAnyOrder(
      BcastConstScalar(0.5), m::Op(src),
      m::AddAnyOrder(BcastConstScalar(1.0), m::Op(&errf).WithOneUser()));

  bool matched = Match(instr, common_pattern);
  if (matched) {
    // The subexpression 0.044715 * x**3 appears in GELU approximate activation.
    // However, it is often optimized by other HLO passes into an expression of
    // 0.044715 * x * (x * x). Since there are three consecutive multiplies,
    // there could be a large number of patterns. We try to capture some of
    // those:
    //
    //      1. (0.044715 * x) * x * x
    //      2. 0.044715 * (x * x) * x
    //
    // Note each of the above could in turn have various patterns due to
    // associativity and commutativity properties of multiply.
    auto subexpr_pattern = m::AnyOf<HloInstruction>(
        MultiplyMultiplyAnyOrder(
            m::MultiplyAnyOrder(BcastConstScalarNear(0.044715),
                                m::Op().Is(*src)),
            m::Op().Is(*src), m::Op().Is(*src)),
        MultiplyMultiplyAnyOrder(
            BcastConstScalarNear(0.044715),
            m::Multiply(m::Op().Is(*src), m::Op().Is(*src)), m::Op().Is(*src)));

    auto errf_apprx_pattern =
        m::Tanh(m::MultiplyAnyOrder(
                    BcastConstScalarNear(sqrt(M_2_PI)),
                    m::AddAnyOrder(m::Op().Is(*src), subexpr_pattern)
                        .WithOneUser()))
            .WithOneUser();

    HloInstruction* erf;
    auto errf_exact_pattern =
        m::Op(&erf)
            .WithOpcode(HloOpcode::kErf)
            .WithOperand(
                0, m::MultiplyAnyOrder(m::Op(src),
                                       m::AnyOf<HloInstruction>(
                                           BcastConstScalarNear(0.707106769),
                                           BcastConstScalarNear(0.70703125),
                                           BcastConstScalarNear(0.707182348)))
                       .WithOneUser())
            .WithOneUser();

    if (Match(errf, errf_apprx_pattern)) {
      // Matched Gelu-approximate pattern
      return OneDnnFusionConfig::GELU_TANH;
    } else if (Match(errf, errf_exact_pattern)) {
      // Matched Gelu-exact pattern
      return OneDnnFusionConfig::GELU_ERF;
    }
  }
  return OneDnnFusionConfig::UNDEFINED;
}

// OneDNN matmul and convolution can fuse add operation with automatic
// broadcasting along the addend's dimensions that are 1s. When compatible,
// Broadcast can be replaced by Bitcast, which is much cheaper. Compute new
// shape for the Bitcast.
absl::StatusOr<Shape> AdjustAddendShape(const HloInstruction* contraction,
                                        const HloInstruction* addend,
                                        const HloInstruction* broadcast_instr) {
  if (!broadcast_instr) {
    // TODO(intel-tf): Modify this condition when Contraction + Bias +
    // Add is enabled.
    if (IsOneDnnConvolutionInstr(contraction) &&
        ShapeUtil::TrueRank(addend->shape()) == 1 &&
        addend->shape().rank() != 1) {
      return ShapeUtil::FilterDimensions(
          [&addend](int64_t dim) {
            return ShapeUtil::GetDimension(addend->shape(), dim) != 1;
          },
          addend->shape());
    }
    return addend->shape();
  }
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

  // If rank(new_shape) > rank(instr), extra dimensions with value = 1 can be
  // deleted from the new_shape.
  auto instr_shape = contraction->shape();
  int64_t rank_difference = new_shape.rank() - instr_shape.rank();
  auto new_dims = new_shape.dimensions();
  std::vector<int64_t> dims_to_delete;
  for (int i = 0; i < rank_difference; ++i) {
    if (new_dims[i] == 1) {
      dims_to_delete.push_back(i);
    }
  }
  new_shape = ShapeUtil::DeleteDimensions(dims_to_delete, new_shape);

  // New shape for bias should satisfy the condition:
  //   rank(new_shape) <= rank(instr).
  if (new_shape.rank() > instr_shape.rank()) {
    return absl::CancelledError(
        "Bias shape could not be adjusted for a fusion.");
  }

  return new_shape;
};

inline bool IsOperandFusible(HloInstruction* operand, HloInstruction* instr) {
  // Check if the operand's shape is compatible for fusion.
  // An operand is fusable if
  //    1. rank(operand) <= rank(instr) and
  //    2. Starting from the last dim in backward direction, the dimension
  //       size of operand is either 1 or same to dot.
  auto operand_dims = operand->shape().dimensions();
  auto instr_dims = instr->shape().dimensions();
  if (operand_dims.size() > instr_dims.size()) return false;
  int operand_idx = operand_dims.size() - 1;
  int instr_idx = instr_dims.size() - 1;
  for (; operand_idx >= 0; --operand_idx, --instr_idx) {
    if (operand_dims[operand_idx] != 1 &&
        operand_dims[operand_idx] != instr_dims[instr_idx])
      return false;
  }
  return true;
}

template <typename Pattern>
inline auto OptionalConvertAndBitcast(HloInstruction** optional_convert,
                                      HloInstruction** optional_bitcast,
                                      Pattern pattern) {
  // Checks the presence of some intermediate operations that can be moved /
  // folded to allow dot fusion with add.
  // Try to match either of the following:
  //   1. pattern-root -> bf16/f16-to-fp32 convert -> bitcast
  //   2. pattern-root -> bf16/f16-to-fp32 convert
  //   3. pattern-root -> bitcast
  //   4. pattern-root
  auto common = m::AnyOf<HloInstruction>(
      pu::SupportedConvert(optional_convert, std::move(pattern).WithOneUser())
          .WithElementType(PrimitiveType::F32),
      std::move(pattern).WithOneUser());
  return m::AnyOf<HloInstruction>(
      BitcastWithReshapeSemantics(optional_bitcast, common), common);
}

}  // namespace

bool OneDnnContractionRewriter::ShouldRewriteDot(
    const HloInstruction* dot_instr, bool before_layout_assignment) {
  if (dot_instr->opcode() != HloOpcode::kDot) return false;
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
  // OneDNN only supports rank <= kOneDnnMaxNDims and singular non-contracting
  // dimensions. We should not rewrite if any of these conditions are violated.
  if (lhs_shape.rank() <= 0 || lhs_shape.rank() > kOneDnnMaxNDims ||
      rhs_shape.rank() <= 0 || rhs_shape.rank() > kOneDnnMaxNDims ||
      output_shape.rank() > std::min({lhs_shape.rank(), rhs_shape.rank(),
                                      static_cast<int64_t>(kOneDnnMaxNDims)})) {
    return false;
  }

  // Layout should be row-major, contraction dimensions captures transpose
  // scenarios in last two dimensions.
  // Col-major layouts are corrected to row-major for BatchDot operation as
  // part of the layout-assignment pass.
  // Skip row-major layout check before layout-assignment pass
  if (!before_layout_assignment) {
    bool row_major = IsRowMajor(lhs_shape) && IsRowMajor(rhs_shape) &&
                     IsRowMajor(output_shape);
    if (!row_major) return false;
  }

  auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
  int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);
  // Supported contraction is only in one of last two dimensions.
  if (lhs_dim_k < lhs_shape.rank() - 2 || rhs_dim_k < rhs_shape.rank() - 2) {
    return false;
  }

  // OneDNN matmul has scratch allocation and copy overheads. The overheads
  // can be amortized if there is sufficient number of flops. We don't rewrite
  // for small cases (determined empirically).
  // TODO(intel-tf): Relax the condition when more optimizations in oneDNN
  // matmul is achieved.
  auto num_flops = xla::HloCostAnalysis::GetDotFlops(lhs_shape, output_shape,
                                                     dot_dim_numbers);
  auto rank = output_shape.rank();
  auto flops_threshold = (rank <= 2) ? (1 << 24) : (1 << 19);
  return (num_flops >= flops_threshold);
}

bool OneDnnContractionRewriter::ShouldRewriteConv(
    const HloInstruction* conv_instr) {
  if (conv_instr->opcode() != HloOpcode::kConvolution) return false;
  if (conv_instr->HasControlDependencies()) return false;
  if (!IsSupportedType(conv_instr->shape().element_type())) return false;
  if (conv_instr->batch_group_count() != 1) return false;

  // TODO(intel-tf): Remove this restriction after enabling backward weights
  // support
  if (conv_instr->operand(1)->opcode() == HloOpcode::kReverse) return false;

  const Shape& inp_shape = conv_instr->operand(0)->shape();
  const Shape& ker_shape = conv_instr->operand(1)->shape();
  const Shape& out_shape = conv_instr->shape();
  if (ShapeUtil::IsZeroElementArray(inp_shape) ||
      ShapeUtil::IsZeroElementArray(ker_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  auto dims = conv_instr->window().dimensions().size();
  if (dims >= 4 || dims <= 0) return false;

  if (inp_shape.rank() != ker_shape.rank() ||
      inp_shape.rank() != out_shape.rank()) {
    return false;
  }

  return true;
}

class OneDnnContractionRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  // Matches patterns for possible MatMul fusions that are supported by oneDNN
  // library. Matched HLO instruction(s) are replaced by custom call.
  absl::Status HandleDot(HloInstruction* instr) override {
    HloInstruction* dot_instr;
    auto pattern = m::Op(&dot_instr).WithOpcode(HloOpcode::kDot);
    if (!Match(instr, pattern)) return absl::OkStatus();

    TF_RETURN_IF_ERROR(
        ValidateDotDimensionNumbers(dot_instr->dot_dimension_numbers()));
    if (!OneDnnContractionRewriter::ShouldRewriteDot(dot_instr)) {
      TF_RETURN_IF_ERROR(UpcastDotToF32(dot_instr));
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(dot_instr, ReconfigureDotDimensions(dot_instr));
    auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
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
    return absl::OkStatus();
  }

  absl::Status HandleConvolution(HloInstruction* conv) override {
    if (!OneDnnContractionRewriter::ShouldRewriteConv(conv)) {
      return absl::OkStatus();
    }

    const Shape& conv_shape = conv->shape();
    auto dims = conv->window().dimensions().size();
    const ConvolutionDimensionNumbers& conv_dims =
        conv->convolution_dimension_numbers();

    BackendConfig backend_config;
    OneDnnConvolutionConfig* conv_config =
        backend_config.mutable_onednn_conv_config();

    conv_config->set_dims(conv_shape.rank());
    conv_config->set_feature_groups(conv->feature_group_count());
    conv_config->mutable_input()->mutable_data()->set_batch_dim(
        conv_dims.input_batch_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_input_feature_dim(
        conv_dims.kernel_input_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_batch_dim(
        conv_dims.output_batch_dimension());
    conv_config->mutable_input()->mutable_data()->set_feature_dim(
        conv_dims.input_feature_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_output_feature_dim(
        conv_dims.kernel_output_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_feature_dim(
        conv_dims.output_feature_dimension());

    const Shape& output_shape = conv->shape();

    for (auto it = conv->window().dimensions().begin();
         it != conv->window().dimensions().end(); it++) {
      if ((*it).padding_low() < 0 || (*it).padding_high() < 0 ||
          (*it).stride() < 0 || (*it).base_dilation() != 1 ||
          (*it).window_reversal()) {
        return absl::OkStatus();
      }
      // Changing the input subspace of uint repeated fields from whole numbers
      // to natural nummbers to avoid misinterpretation of buffer values.
      conv_config->mutable_window()->add_pad_left((*it).padding_low() + 1);
      conv_config->mutable_window()->add_pad_right((*it).padding_high() + 1);
      conv_config->mutable_window()->add_strides((*it).stride() + 1);
      conv_config->mutable_window()->add_window_dilations(
          (*it).window_dilation() + 1);
    }

    for (int i = 0; i < dims; i++) {
      conv_config->mutable_input()->mutable_data()->add_spatial_dims(
          conv_dims.input_spatial_dimensions()[i] + 1);
      conv_config->mutable_kernel()->mutable_filter()->add_spatial_dims(
          conv_dims.kernel_spatial_dimensions()[i] + 1);
      conv_config->mutable_output()->mutable_data()->add_spatial_dims(
          conv_dims.output_spatial_dimensions()[i] + 1);
    }

    HloInstruction* custom_call =
        conv->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {conv->mutable_operand(0), conv->mutable_operand(1)},
            "__onednn$convolution"));

    TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(conv, custom_call));
    return absl::OkStatus();
  }

  absl::Status HandleAdd(HloInstruction* instr) override {
    // Try to fuse Add to the instr. However,
    // HLO Add instruction might receive the addends after additional
    // processing like Broadcast, Bitcast, Convert, etc. is applied to the raw
    // addends. Here, the following possible pattern is matched.
    //
    // clang-format off
    //
    //         Dot / Conv                           addend
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

    HloInstruction *addend_intermediate, *contraction;
    HloInstruction* optional_contraction_bitcast = nullptr;
    HloInstruction* optional_contraction_convert = nullptr;

    auto pattern = m::AddAnyOrder(
        &instr,
        OptionalConvertAndBitcast(&optional_contraction_convert,
                                  &optional_contraction_bitcast,
                                  OneDnnFusibleInstr(&contraction))
            .WithOneUser(),
        m::Op(&addend_intermediate));

    if (Match(instr, pattern)) {
      if (!IsSupportedType(contraction->shape().element_type()))
        return absl::OkStatus();
      // TODO(intel-tf): Remove the condition below when the fusion Contraction
      // + Add(bias) + Add(e.g., residual) is enabled.
      auto contraction_config = contraction->backend_config<BackendConfig>();
      auto orig_fusion_config = GetFusionsConfig(&contraction_config);
      if (!orig_fusion_config->ops().empty() &&
          orig_fusion_config->ops(0) == OneDnnFusionConfig::BIAS) {
        return absl::OkStatus();
      }
      std::vector<HloInstruction*> new_operands;
      for (auto operand : contraction->operands()) {
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
      // is compatible for onednn fusion.
      HloInstruction* addend = nullptr;
      HloInstruction* optional_addend_broadcast = nullptr;
      auto addend_pattern = m::AnyOf<HloInstruction>(
          m::Broadcast(&optional_addend_broadcast,
                       m::Convert(&addend, m::Op())),
          m::Convert(m::Broadcast(&optional_addend_broadcast, m::Op(&addend))),
          m::Convert(&addend, m::Op()),
          m::Broadcast(&optional_addend_broadcast, m::Op(&addend)),
          m::Op(&addend));
      if (!Match(addend_intermediate, addend_pattern)) return absl::OkStatus();

      // oneDNN library requires Convolution biases to always have rank 1.
      // Therefore, these bias shapes should remain unchanged.
      if (IsOneDnnMatmulInstr(contraction) || addend->shape().rank() != 1) {
        auto new_shape =
            AdjustAddendShape(contraction, addend, optional_addend_broadcast);
        if (!new_shape.ok()) {
          VLOG(2) << new_shape.status();
          return absl::OkStatus();
        } else if (!ShapeUtil::Equal(*new_shape, addend->shape())) {
          addend = addend->AddInstruction(
              HloInstruction::CreateBitcast(new_shape.value(), addend));
        }
      }

      // Validate addend for fusion.
      if (IsSupportedType(addend->shape().element_type()) &&
          IsOperandFusible(addend, contraction)) {
        new_operands.push_back(addend);
      } else {
        return absl::OkStatus();
      }

      auto custom_call = Cast<HloCustomCallInstruction>(
          instr->AddInstruction(contraction->CloneWithNewOperands(
              contraction->shape(), new_operands)));

      auto backend_config = custom_call->backend_config<BackendConfig>();
      auto fusions_config = GetFusionsConfig(&backend_config);
      auto optimization_config = GetOptimizationsConfig(&backend_config);
      // TODO(intel-tf): Here, we allow 1D addends only when they are the first
      // fused op. Remove this restriction once oneDNN has an optimized
      // implementation for broadcasted add across all dimensions.
      OneDnnFusionConfig_FusionKind kind =
          (ShapeUtil::TrueRank(addend->shape()) == 1)
              ? (fusions_config->ops().empty() ? OneDnnFusionConfig::BIAS
                                               : OneDnnFusionConfig::UNDEFINED)
              : OneDnnFusionConfig::BINARY_ADD;
      if (kind == OneDnnFusionConfig::UNDEFINED) return absl::OkStatus();

      fusions_config->add_ops(kind);

      if (optional_addend_broadcast) {
        optimization_config->set_bias_broadcast(true);
      }
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(*backend_config));

      HloInstruction* new_instr;
      // If matched pattern has custom-call -> bitcast -> add, then we need to
      // insert bitcast after the new fusion to maintain the correct shape
      // (new-custom-call -> bitcast). Also, this will optionally be followed
      // by -> convert for bf16 case to avoid datatype mismatch.
      if (optional_contraction_bitcast != nullptr &&
          optional_contraction_bitcast->opcode() == HloOpcode::kBitcast) {
        if (optional_contraction_convert != nullptr &&
            optional_contraction_convert->opcode() == HloOpcode::kConvert) {
          auto bitcast_call =
              custom_call->AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::ChangeElementType(
                      instr->shape(), custom_call->shape().element_type()),
                  custom_call));
          new_instr =
              bitcast_call->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(
                      bitcast_call->shape(),
                      optional_contraction_convert->shape().element_type()),
                  bitcast_call));
        } else {
          new_instr = custom_call->AddInstruction(
              HloInstruction::CreateBitcast(instr->shape(), custom_call));
        }
      } else {
        if (optional_contraction_convert != nullptr &&
            optional_contraction_convert->opcode() == HloOpcode::kConvert) {
          new_instr = custom_call->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(
                  custom_call->shape(),
                  optional_contraction_convert->shape().element_type()),
              custom_call));
        } else {
          new_instr = custom_call;
        }
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }
    return absl::OkStatus();
  }

  absl::Status HandleMaximum(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr,
              m::MaximumAnyOrder(ElementwiseSafeIntermediates(
                                     &intermediate_instr, &optional_bitcast,
                                     OneDnnFusibleInstr(&contraction))
                                     .WithOneUser(),
                                 BcastConstScalar(0)))) {
      return FuseActivation(OneDnnFusionConfig::RELU, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  auto ELUActivation(HloInstruction* instr, HloInstruction** src) {
    //  Reference: tensorflow/compiler/tf2xla/kernels/elu_op.cc
    //  const auto zero = ScalarLike(x, 0);
    //  const auto pred = Gt(x, zero);
    //  const auto expm1 = Expm1(x);
    //  return Select(pred, x, expm1);
    auto pattern = m::Select(
        m::Gt(pu::OptionalConvert(m::Op(src)), BcastConvertConstScalar(0)),
        m::Op(src),
        pu::OptionalConvert(m::Expm1(pu::OptionalConvert(m::Op(src)))));
    return Match(instr, pattern);
  }

  absl::Status HandleSelect(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    HloInstruction* src;
    // Attempt to elide ELU subgraph and fuse ELU activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (ELUActivation(instr, &src)) {
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction)))) {
        return FuseActivation(OneDnnFusionConfig::ELU, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleTanh(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide Tanh and fuse Tanh activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr, m::Tanh(ElementwiseSafeIntermediates(
                                 &intermediate_instr, &optional_bitcast,
                                 OneDnnFusibleInstr(&contraction))
                                 .WithOneUser()))) {
      return FuseActivation(OneDnnFusionConfig::TANH, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  absl::Status HandleClamp(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide RELU6 and fuse RELU6 activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr, m::Clamp(BcastConstScalar(0),
                              ElementwiseSafeIntermediates(
                                  &intermediate_instr, &optional_bitcast,
                                  OneDnnFusibleInstr(&contraction))
                                  .WithOneUser(),
                              BcastConstScalar(6)))) {
      return FuseActivation(OneDnnFusionConfig::RELU6, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  absl::Status HandleMultiply(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* src;
    auto activation = GELUActivation(instr, &src);
    if (activation != OneDnnFusionConfig::UNDEFINED) {
      HloInstruction* optional_bitcast = nullptr;
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction)))) {
        return FuseActivation(activation, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }

    HloInstruction* constant;
    HloInstruction* optional_convert = nullptr;
    auto pattern =
        m::Op(&instr)
            .WithOpcode(HloOpcode::kMultiply)
            .WithBinaryOperandsAnyOrder(
                m::AnyOf<HloInstruction>(
                    pu::SupportedConvert(&optional_convert,
                                         OneDnnFusibleInstr(&contraction))
                        .WithElementType(PrimitiveType::F32),
                    OneDnnFusibleInstr(&contraction))
                    .WithOneUser(),
                m::Broadcast(m::Constant(&constant)));

    if (Match(instr, pattern)) {
      std::vector<HloInstruction*> new_operands;
      auto constant_value = GetConstantValueAsFloat32(constant);
      if (!constant_value) {
        return absl::OkStatus();
      }

      for (auto operand : contraction->operands()) {
        new_operands.push_back(operand);
      }
      auto custom_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          contraction->CloneWithNewOperands(instr->shape(), new_operands)));
      auto backend_config = custom_call->backend_config<BackendConfig>();
      auto fusions_config = GetFusionsConfig(&backend_config);
      fusions_config->add_ops(OneDnnFusionConfig::LINEAR);
      // Casting to int32 because of issues in proto config for decimal types
      // handling.
      fusions_config->set_alpha_typecast(
          *(reinterpret_cast<int32_t*>(&constant_value.value())));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(*backend_config));
      HloInstruction* new_instr;
      if (optional_convert != nullptr &&
          optional_convert->opcode() == HloOpcode::kConvert) {
        new_instr = custom_call->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(
                custom_call->shape(), optional_convert->shape().element_type()),
            custom_call));
      } else {
        new_instr = custom_call;
      }

      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }
    return absl::OkStatus();
  }

  auto SigmoidActivation(HloInstruction* instr, HloInstruction** src) {
    return Match(instr,
                 m::Divide(BcastConstScalar(1.0),
                           m::AddAnyOrder(BcastConstScalar(1.0),
                                          m::Exp(m::Negate(m::Op(src))))));
  }

  absl::Status HandleDivide(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    HloInstruction* src;
    if (SigmoidActivation(instr, &src)) {
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction))
                         .WithOneUser())) {
        return FuseActivation(OneDnnFusionConfig::SIGMOID, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }
    return absl::OkStatus();
  }

  absl::Status FuseActivation(OneDnnFusionConfig_FusionKind kind,
                              HloInstruction* activation,
                              HloInstruction* contraction,
                              HloInstruction* intermediate_instr = nullptr,
                              HloInstruction* optional_bitcast = nullptr) {
    auto backend_config = contraction->backend_config<BackendConfig>();
    auto fusions_config = GetFusionsConfig(&backend_config);
    fusions_config->add_ops(kind);
    TF_RETURN_IF_ERROR(contraction->set_backend_config(*backend_config));
    std::unique_ptr<HloInstruction> output = contraction->Clone();
    if (optional_bitcast != nullptr &&
        optional_bitcast->opcode() == HloOpcode::kBitcast) {
      HloInstruction* new_instr = nullptr;
      if (intermediate_instr != nullptr &&
          intermediate_instr->opcode() == HloOpcode::kConvert) {
        auto bitcast_call =
            contraction->AddInstruction(HloInstruction::CreateBitcast(
                ShapeUtil::ChangeElementType(
                    optional_bitcast->shape(),
                    contraction->shape().element_type()),
                contraction));
        new_instr = bitcast_call->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(
                bitcast_call->shape(),
                intermediate_instr->shape().element_type()),
            bitcast_call));
        return ReplaceInstruction(activation, new_instr);
      }
    } else if (intermediate_instr) {
      output = intermediate_instr->CloneWithNewOperands(
          intermediate_instr->shape(),
          {contraction->parent()->AddInstruction(std::move(output))});
    }

    return ReplaceWithNewInstruction(activation, std::move(output));
  }

  // This function changes dot instruction for supported matrix
  // multiplication scenarios. In particular, it changes the shape
  // of lhs, rhs and result arrays.
  //    - lhs configuration scenario
  //      lhs:    [batch_dims,contracting_dim] to [batch_dims,1,contracting_dim]
  //      result: [batch_dims,feature_dim] to [batch_dims,1,feature_dim]
  //
  //    - rhs configuration scenario
  //      rhs:    [batch_dims,contracting_dim] to [batch_dims,contracting_dim,1]
  //      result: [batch_dims,feature_dim] to [batch_dims,feature_dim, 1]
  //
  //    - both lhs and rhs configuration scenario
  //      lhs:    [batch_dims,contracting_dim] to [batch_dims,1,contracting_dim]
  //      rhs:    [batch_dims,contracting_dim] to [batch_dims,contracting_dim,1]
  //      result: [batch_dims] to [batch_dims,1,1]
  absl::StatusOr<HloInstruction*> ReconfigureDotDimensions(
      HloInstruction* dot_instr) {
    HloInstruction* lhs = dot_instr->mutable_operand(0);
    HloInstruction* rhs = dot_instr->mutable_operand(1);
    DotDimensionNumbers dim_numbers = dot_instr->dot_dimension_numbers();

    auto lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
    auto lhs_contraction_dims = dim_numbers.lhs_contracting_dimensions();
    bool is_lhs_vector = lhs->shape().rank() ==
                         (lhs_batch_dims.size() + lhs_contraction_dims.size());

    auto rhs_batch_dims = dim_numbers.rhs_batch_dimensions();
    auto rhs_contraction_dims = dim_numbers.rhs_contracting_dimensions();
    bool is_rhs_vector = rhs->shape().rank() ==
                         (rhs_batch_dims.size() + rhs_contraction_dims.size());

    if (!is_lhs_vector && !is_rhs_vector) return dot_instr;

    std::vector<int64_t> adjusted_lhs_dims(lhs->shape().dimensions().begin(),
                                           lhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_rhs_dims(rhs->shape().dimensions().begin(),
                                           rhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_dot_dims(
        dot_instr->shape().dimensions().begin(),
        dot_instr->shape().dimensions().end());

    if (is_lhs_vector) {
      auto lhs_it = adjusted_lhs_dims.begin() + lhs_batch_dims.size();
      adjusted_lhs_dims.insert(lhs_it, 1, 1);
      auto result_it = adjusted_dot_dims.begin() + lhs_batch_dims.size();
      adjusted_dot_dims.insert(result_it, 1, 1);
      auto lhs_contraction_dim =
          dot_instr->dot_dimension_numbers().lhs_contracting_dimensions(0);
      dim_numbers.set_lhs_contracting_dimensions(0, lhs_contraction_dim + 1);
      lhs = lhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(lhs->shape().element_type(), adjusted_lhs_dims),
          lhs));
    }

    if (is_rhs_vector) {
      auto it = adjusted_rhs_dims.end();
      adjusted_rhs_dims.insert(it, 1, 1);
      auto result_it = adjusted_dot_dims.end();
      adjusted_dot_dims.insert(result_it, 1, 1);
      rhs = rhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(rhs->shape().element_type(), adjusted_rhs_dims),
          rhs));
    }

    HloInstruction* adjusted_dot =
        dot_instr->AddInstruction(HloInstruction::CreateDot(
            ShapeUtil::MakeShape(dot_instr->shape().element_type(),
                                 adjusted_dot_dims),
            lhs, rhs, dim_numbers, dot_instr->precision_config()));

    HloInstruction* replacement_instr = adjusted_dot->AddInstruction(
        HloInstruction::CreateBitcast(dot_instr->shape(), adjusted_dot));

    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, replacement_instr));
    return adjusted_dot;
  }

  // This function upcasts BF16 dots to F32 if we are unable to rewrite them to
  // oneDNN custom calls.
  absl::Status UpcastDotToF32(HloInstruction* dot_instr) {
    if (dot_instr->shape().element_type() != BF16) return absl::OkStatus();
    std::vector<HloInstruction*> new_operands;
    auto bf16_operands = dot_instr->operands();

    std::for_each(
        bf16_operands.begin(), bf16_operands.end(),
        [&new_operands](HloInstruction* instr) {
          new_operands.push_back(
              instr->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(instr->shape(), F32), instr)));
        });

    HloInstruction* f32_dot =
        dot_instr->AddInstruction(dot_instr->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(dot_instr->shape(), F32),
            new_operands));

    HloInstruction* replacement_instr =
        f32_dot->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(f32_dot->shape(), BF16), f32_dot));

    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, replacement_instr));
    return absl::OkStatus();
  }
};

class OneDnnPostRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  OneDnnPostRewriteVisitor(int intra_op_parallelism,
                           const tsl::thread::ThreadPool* compile_threadpool)
      : intra_op_parallelism_(intra_op_parallelism > 0
                                  ? intra_op_parallelism
                                  : tsl::port::MaxParallelism()),
        evaluator_(/*max_loop_iterations=*/0) {
    if (compile_threadpool) {
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(compile_threadpool->AsEigenThreadPool(),
                                      compile_threadpool->NumThreads()));
    } else {
      threadpool_handle_.reset(new tsl::thread::ThreadPool(
          tsl::Env::Default(), "XLACpuCompile", tsl::port::MaxParallelism()));
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(threadpool_handle_->AsEigenThreadPool(),
                                      threadpool_handle_->NumThreads()));
    }

#ifndef ENABLE_ONEDNN_OPENMP
    // Set oneDNN concurrency settings (which is thread-local)
    tsl::OneDnnThreadPool::set_onednn_max_threads(intra_op_parallelism_);
#endif
  }

  absl::Status HandleCustomCall(HloInstruction* custom_call) override {
    HloInstruction* contraction;
    if (Match(custom_call, OneDnnMatmulInstr(&contraction))) {
      return HandleCustomCallInternal<dnnl::matmul::primitive_desc>(
          custom_call);
    } else if (Match(custom_call, OneDnnConvolutionInstr(&contraction))) {
      return HandleCustomCallInternal<
          dnnl::convolution_forward::primitive_desc>(custom_call);
    }
    return DefaultAction(custom_call);
  }

  template <typename PrimDesc>
  absl::Status HandleCustomCallInternal(HloInstruction* custom_call) {
    auto scratch_add = AddScratch<PrimDesc>(custom_call);
    if (scratch_add.ok()) {
      custom_call = *scratch_add;
    } else {
      VLOG(2) << scratch_add.status();
    }
    // TODO(intel-tf): Remove this condition after enabling weights prepacking
    // for convolutions
    if constexpr (std::is_same_v<PrimDesc, dnnl::matmul::primitive_desc>) {
      auto weights_prepack = PrepackWeights<PrimDesc>(custom_call);
      if (!weights_prepack.ok()) {
        VLOG(2) << weights_prepack.status();
      }
    }
    return absl::OkStatus();
  }

  template <typename>
  absl::Status SetWeightsPrepack(HloInstruction*, bool);

  template <typename>
  absl::Status SetUserScratch(HloInstruction*, bool);

  template <typename>
  bool GetWeightsPrepack(HloInstruction*);

  template <typename>
  bool GetUserScratch(HloInstruction*);

  // Add scratch for matmul and convolution by changing the result of
  // custom-call to tuple(result, scratch)
  template <typename PrimDesc>
  absl::StatusOr<HloInstruction*> AddScratch(HloInstruction* custom_call) {
    if (GetUserScratch<PrimDesc>(custom_call)) {
      return custom_call;
    }
    TF_RETURN_IF_ERROR(SetUserScratch<PrimDesc>(custom_call, true));
    auto prim_desc = CreateOneDnnPrimDesc<PrimDesc>(custom_call);
    int64_t scratch_size = prim_desc->scratchpad_desc().get_size();
    Shape scratch_shape = ShapeUtil::MakeShape(U8, {scratch_size});
    Shape tuple_shape =
        ShapeUtil::MakeTupleShape({custom_call->shape(), scratch_shape});
    auto new_custom_call = custom_call->AddInstruction(
        custom_call->CloneWithNewShape(tuple_shape));
    HloInstruction* gte =
        new_custom_call->AddInstruction(HloInstruction::CreateGetTupleElement(
            custom_call->shape(), new_custom_call, 0));
    auto status = ReplaceInstruction(custom_call, gte);
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(SetUserScratch<PrimDesc>(custom_call, false));
      return absl::CancelledError("Adding scratch is unsuccessful.");
    }
    return new_custom_call;
  }

  template <typename PrimDesc>
  absl::StatusOr<HloInstruction*> PrepackWeights(HloInstruction* custom_call) {
    if (GetWeightsPrepack<PrimDesc>(custom_call)) {
      return custom_call;
    }
    auto weights = custom_call->operand(1);
    auto weights_shape = weights->shape();
    Literal weights_literal;
    if (!(weights_shape.rank() == 2 &&
          evaluator_.TryEvaluate(weights, &weights_literal, true))) {
      return absl::CancelledError(
          "Cannot prepack weights. Not constant 2D weights.");
    }
    auto plain_weights_md = ShapeToMemDesc(weights_shape);
    if constexpr (std::is_same<PrimDesc, dnnl::matmul::primitive_desc>::value) {
      TF_ASSIGN_OR_RETURN(auto backend_config,
                          custom_call->backend_config<BackendConfig>());
      TRANSPOSE_LAST_TWO_DIMS_IF(
          backend_config.onednn_matmul_config().transpose_b(),
          plain_weights_md);
    }
    TF_RETURN_IF_ERROR(SetWeightsPrepack<PrimDesc>(custom_call, true));
    auto prim_desc = CreateOneDnnPrimDesc<PrimDesc>(custom_call);
    auto packed_weights_md = prim_desc->weights_desc();
    auto packed_weights_shape = MemDescToXlaShapeFlattened(packed_weights_md);
    auto packed_weights_literal = Literal(packed_weights_shape);
    ReorderWeight(plain_weights_md, weights_literal.untyped_data(),
                  packed_weights_md, packed_weights_literal.untyped_data());
    HloInstruction* reordered_weight = custom_call->AddInstruction(
        HloInstruction::CreateConstant(std::move(packed_weights_literal)));
    auto status =
        custom_call->ReplaceOperandWithDifferentShape(1, reordered_weight);
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(SetWeightsPrepack<PrimDesc>(custom_call, false));
      return absl::CancelledError(
          "Cannot replace plain weights with prepacked weights.");
    } else {
      return custom_call;
    }
  }

  void ReorderWeight(const dnnl::memory::desc& src_md, void* src_buf,
                     const dnnl::memory::desc& dst_md, void* dst_buf) {
    auto onednn_threadpool = CreateOneDnnThreadPool(threadpool_device_.get());
    dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
    auto onednn_stream = MakeOneDnnStream(cpu_engine, onednn_threadpool.get());
    auto src_mem = dnnl::memory(src_md, cpu_engine, src_buf);
    auto dst_mem = dnnl::memory(dst_md, cpu_engine, dst_buf);
    dnnl::reorder reorder_prim{src_mem, dst_mem};
    reorder_prim.execute(onednn_stream, src_mem, dst_mem);
    onednn_stream.wait();
  }

 private:
  int intra_op_parallelism_;
  HloEvaluator evaluator_;
  std::unique_ptr<tsl::thread::ThreadPool> threadpool_handle_;
  std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
};

#define EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GETTER, PRIM_DESC, CONFIG,      \
                                               SUB_CONFIG, FIELD)              \
  template <>                                                                  \
  inline bool OneDnnPostRewriteVisitor::GETTER<PRIM_DESC>(HloInstruction *     \
                                                          custom_call) {       \
    auto backend_config = custom_call->backend_config<BackendConfig>();        \
    return backend_config.ok() ? backend_config->CONFIG().SUB_CONFIG().FIELD() \
                               : false;                                        \
  }

EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GetUserScratch,
                                       dnnl::matmul::primitive_desc,
                                       onednn_matmul_config,
                                       optimization_config, user_scratchpad);
EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GetWeightsPrepack,
                                       dnnl::matmul::primitive_desc,
                                       onednn_matmul_config,
                                       optimization_config, weights_prepacked);
EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(
    GetUserScratch, dnnl::convolution_forward::primitive_desc,
    onednn_conv_config, optimization_config, user_scratchpad);

#define EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SETTER, PRIM_DESC, CONFIG_TYPE, \
                                               CONFIG, SUB_CONFIG, FIELD)      \
  template <>                                                                  \
  inline absl::Status OneDnnPostRewriteVisitor::SETTER<PRIM_DESC>(             \
      HloInstruction * custom_call, bool value) {                              \
    TF_ASSIGN_OR_RETURN(auto backend_config,                                   \
                        custom_call->backend_config<BackendConfig>());         \
    CONFIG_TYPE* config = backend_config.mutable_##CONFIG();                   \
    config->mutable_##SUB_CONFIG()->set_##FIELD(value);                        \
    return custom_call->set_backend_config(backend_config);                    \
  }

EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SetWeightsPrepack,
                                       dnnl::matmul::primitive_desc,
                                       OneDnnMatMulConfig, onednn_matmul_config,
                                       optimization_config, weights_prepacked);
EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SetUserScratch,
                                       dnnl::matmul::primitive_desc,
                                       OneDnnMatMulConfig, onednn_matmul_config,
                                       optimization_config, user_scratchpad);
EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(
    SetUserScratch, dnnl::convolution_forward::primitive_desc,
    OneDnnConvolutionConfig, onednn_conv_config, optimization_config,
    user_scratchpad);

absl::StatusOr<bool> OneDnnContractionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "OneDnnContractionRewriter::Run(), before:\n" + module->ToString());
  OneDnnContractionRewriteVisitor visitor;
  TF_ASSIGN_OR_RETURN(auto result,
                      visitor.RunOnModule(module, execution_threads));

  OneDnnPostRewriteVisitor reorder_visitor(intra_op_parallelism_,
                                           compile_threadpool_);
  TF_ASSIGN_OR_RETURN(auto result2,
                      reorder_visitor.RunOnModule(module, execution_threads));
  XLA_VLOG_LINES(
      3, "OneDnnContractionRewriter::Run(), after:\n" + module->ToString());
  return {result || result2};
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
