/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cudnn/cudnn_version.h"
#include "xla/codegen/emitters/computation_fingerprint.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

namespace fe = cudnn_frontend;
namespace graph = fe::graph;

inline std::optional<fe::PointwiseMode_t> GetElementwiseMode(
    const HloInstruction& instruction) {
  const HloOpcode opcode = instruction.opcode();
  using m = fe::PointwiseMode_t;
  switch (opcode) {
    case HloOpcode::kAbs:
      return m::ABS;
    case HloOpcode::kAdd:
      return m::ADD;
    case HloOpcode::kCeil:
      return m::CEIL;
    case HloOpcode::kCompare:
      switch (instruction.comparison_direction()) {
        case Comparison::Direction::kEq:
          return m::CMP_EQ;
        case Comparison::Direction::kNe:
          return m::CMP_NEQ;
        case Comparison::Direction::kGe:
          return m::CMP_GE;
        case Comparison::Direction::kGt:
          return m::CMP_GT;
        case Comparison::Direction::kLe:
          return m::CMP_LE;
        case Comparison::Direction::kLt:
          return m::CMP_LT;
      }
      break;
    case HloOpcode::kConvert:
      return m::IDENTITY;
    case HloOpcode::kCos:
      return m::COS;
    case HloOpcode::kDivide:
      return m::DIV;
    case HloOpcode::kExp:
      return m::EXP;
    case HloOpcode::kFloor:
      return m::FLOOR;
    case HloOpcode::kLog:
      return m::LOG;
    case HloOpcode::kMaximum:
      return m::MAX;
    case HloOpcode::kMinimum:
      return m::MIN;
    case HloOpcode::kMultiply:
      return m::MUL;
    case HloOpcode::kNegate:
      return m::NEG;
    case HloOpcode::kPower:
      return m::POW;
    case HloOpcode::kRsqrt:
      return m::RSQRT;
#if CUDNN_VERSION >= 90100
    case HloOpcode::kSelect:
      return m::BINARY_SELECT;
#endif  // CUDNN_VERSION
    case HloOpcode::kSin:
      return m::SIN;
    case HloOpcode::kSqrt:
      return m::SQRT;
    case HloOpcode::kSubtract:
      return m::SUB;
    case HloOpcode::kTan:
      return m::TAN;
    case HloOpcode::kTanh:
      return m::TANH_FWD;
    default:
      return std::nullopt;
  }
}

inline std::optional<fe::DataType_t> ToCudnnDataType(const PrimitiveType type) {
  using t = fe::DataType_t;
  switch (type) {
    case PrimitiveType::F32:
      return t::FLOAT;
    case PrimitiveType::F16:
      return t::HALF;
    case PrimitiveType::BF16:
      return t::BFLOAT16;
    case PrimitiveType::S32:
      return t::INT32;
    case PrimitiveType::S4:
      return t::INT4;
    case PrimitiveType::S8:
      return t::INT8;
    case PrimitiveType::PRED:
      return t::INT8;
    case PrimitiveType::F8E5M2:
      return t::FP8_E5M2;
    case PrimitiveType::F8E4M3FN:
      return t::FP8_E4M3;
    case PrimitiveType::F8E8M0FNU:
      return t::FP8_E8M0;
    case PrimitiveType::F4E2M1FN:
      return t::FP4_E2M1;
    default:
      return std::nullopt;
  }
}

inline std::optional<fe::DataType_t> GetComputeDataType(
    const PrimitiveType type) {
  fe::DataType_t compute_dtype = fe::DataType_t::FLOAT;
  if (type == F64) {
    compute_dtype = fe::DataType_t::DOUBLE;
  } else if (primitive_util::IsIntegralType(type)) {
#if CUDNN_VERSION >= 90100
    compute_dtype = fe::DataType_t::INT32;
#else
    VLOG(3) << "Integer math requires cuDNN 9.1+.";
    return std::nullopt;
#endif  // CUDNN_VERSION
  }
  return compute_dtype;
}

// Extracts dimensions and strides from HLO tensors in the format expected by
// cuDNN.
struct Result {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::optional<std::vector<std::pair<int64_t, int64_t>>> slices;
};

class GemmDimensionAdapter {
  explicit GemmDimensionAdapter(const HloInstruction& dot,
                                TritonFusionAnalysis analysis)
      : analysis_(std::move(analysis)), dot_(dot) {}

 public:
  const TritonFusionAnalysis analysis_;

  static absl::StatusOr<std::optional<GemmDimensionAdapter>> Create(
      const HloComputation& computation) {
    const HloInstruction* maybe_scaled_dot =
        hlo_query::GetFirstInstructionWithOpcode(computation,
                                                 HloOpcode::kScaledDot);
    const HloInstruction* maybe_dot =
        hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot);
    if (maybe_scaled_dot == nullptr && maybe_dot == nullptr) {
      VLOG(3) << "Not a GEMM fusion.";
      return std::nullopt;
    }
    const HloInstruction* dot =
        maybe_dot != nullptr ? maybe_dot : maybe_scaled_dot;
    if (absl::c_any_of(dot->precision_config().operand_precision(),
                       [](int x) { return x != PrecisionConfig::DEFAULT; })) {
      VLOG(3) << "Non-default precision is not supported.";
      return std::nullopt;
    }
    if (dot->precision_config().algorithm() != PrecisionConfig::ALG_UNSET) {
      VLOG(3) << "Non-default algorithm is not supported.";
      return std::nullopt;
    }
    TF_ASSIGN_OR_RETURN(auto analysis,
                        TritonFusionAnalysis::Execute(computation));
    return GemmDimensionAdapter{*dot, std::move(analysis)};
  }

  std::optional<Result> DimensionsAndStrides(
      const HloInstruction& hlo, const TritonFusionAnalysis::Scope scope) {
    const DotDimensionNumbers& dims = dot_.dot_dimension_numbers();
    // GEMM fusions require a specific canonical order of dimensions.
    constexpr int kBatchDimensionIndex = 0;
    constexpr int kOutputLHSNonContractingDimensionIndex = 1;
    std::vector<int64_t> dim_indices;
    int lhs_noncontracting_index = -1;
    switch (scope) {
      case TritonFusionAnalysis::Scope::LHS:
      case TritonFusionAnalysis::Scope::LHS_SCALE:
        lhs_noncontracting_index =
            GetNonContractingDims(dot_.operand(0)->shape(),
                                  dims.lhs_batch_dimensions(),
                                  dims.lhs_contracting_dimensions())
                .value()[0];
        dim_indices = {
            dims.lhs_batch_dimensions().empty() ? -1
                                                : dims.lhs_batch_dimensions(0),
            lhs_noncontracting_index, dims.lhs_contracting_dimensions(0)};
        break;
      case TritonFusionAnalysis::Scope::RHS:
      case TritonFusionAnalysis::Scope::RHS_SCALE:
        dim_indices = {dims.rhs_batch_dimensions().empty()
                           ? -1
                           : dims.rhs_batch_dimensions(0),
                       dims.rhs_contracting_dimensions(0),
                       GetNonContractingDims(dot_.operand(1)->shape(),
                                             dims.rhs_batch_dimensions(),
                                             dims.rhs_contracting_dimensions())
                           .value()[0]};
        break;
      case TritonFusionAnalysis::Scope::OUTPUT:
        lhs_noncontracting_index = dot_.shape().dimensions().size() - 2;
        dim_indices = {dims.lhs_batch_dimensions().empty() ? -1 : 0,
                       lhs_noncontracting_index,
                       dot_.shape().dimensions_size() - 1};
        break;
    }

    Result result;
    result.sizes.reserve(dim_indices.size());
    result.strides.reserve(dim_indices.size());
    result.slices = std::vector<std::pair<int64_t, int64_t>>{};
    result.slices->reserve(dim_indices.size());
    bool slicing_is_present = false;

    for (const int index : dim_indices) {
      const auto* spec = analysis_.IterSpec(scope, &hlo, index);
      if (spec == nullptr) {
        result.sizes.push_back(1);
        result.strides.push_back(
            result.strides.empty() ? 1 : result.strides.back());
        result.slices->push_back({0, 1});
        continue;
      } else {
        if (spec->size() == 1) {
          // The dimension is not split, nothing to do.
        } else if (spec->size() == 2) {
          if (!dims.lhs_batch_dimensions().empty()) {
            VLOG(8) << "Noncontracting dimension split is not compatible with "
                       "batch dimensions.";
            return std::nullopt;
          }
          if (index != lhs_noncontracting_index) {
            VLOG(8) << "Only LHS noncontracting dimension can be split.";
            return std::nullopt;
          }
          switch (scope) {
            case TritonFusionAnalysis::Scope::LHS:
              lhs_noncontracting_split_ = spec->back().count;
              break;
            case TritonFusionAnalysis::Scope::OUTPUT:
              if (lhs_noncontracting_split_ != spec->back().count) {
                VLOG(8) << "Output non-contracting dimension has to be split "
                           "the same way as the LHS input one if it is split.";
                return std::nullopt;
              }
              break;
            default:
              VLOG(8) << "Only LHS noncontracting dimension can be split.";
              return std::nullopt;
          }
          // Assign the major part of the noncontracting dimension to the
          // unused batch one.
          CHECK_EQ(result.sizes[kBatchDimensionIndex], 1);
          result.sizes[kBatchDimensionIndex] = spec->back().count;
          result.strides[kBatchDimensionIndex] = spec->back().stride;
        } else {
          VLOG(8) << "The dimension is split multiple times.";
          return std::nullopt;
        }
        result.sizes.push_back(spec->front().count);
        result.strides.push_back(spec->front().stride);
        result.slices->push_back(
            {spec->front().slice_start,
             spec->front().slice_start + spec->front().sliced_count});
        if (spec->front().count != spec->front().sliced_count) {
          slicing_is_present = true;
        }
      }
    }
    if (lhs_noncontracting_split_ > 1 &&
        scope == TritonFusionAnalysis::Scope::OUTPUT &&
        result.sizes[kBatchDimensionIndex] == 1) {
      // LHS input noncontracting dimension is split but the corresponding
      // output one is not. Assign part of the output one to the unused batch
      // dimension.
      result.sizes[kBatchDimensionIndex] = lhs_noncontracting_split_;
      result.sizes[kOutputLHSNonContractingDimensionIndex] /=
          lhs_noncontracting_split_;
      result.strides[kBatchDimensionIndex] =
          result.strides[kOutputLHSNonContractingDimensionIndex] *
          result.sizes[kOutputLHSNonContractingDimensionIndex];
    }

    // 0 (kBatchDimensionIndex) is always the batch dimension;
    // 1 and 2 are the non-batch ones. cuDNN relies on strides to determine
    // layouts and gets confused when both strides of non-batch dimensions
    // are equal to 1 - this is the case for tensors with 1-sized dimension
    // like [A,1]. The stride of the 1-sized dimension does not matter for
    // correctness because there is no iteration along this dimension, but
    // setting it to A and representing the tensor as its equivalent [1,A]
    // helps cuDNN.
    if (result.strides[1] == 1 && result.strides[2] == 1) {
      const int one_sized_dim_idx = (result.sizes[1] == 1) ? 1 : 2;
      result.strides[one_sized_dim_idx] = result.sizes[1] * result.sizes[2];
    }

    if (!slicing_is_present) {
      result.slices.reset();
    }
    return result;
  }

 private:
  int64_t lhs_noncontracting_split_ = 1;
  const HloInstruction& dot_;
};

using ConvKind = HloConvolutionInstruction::ConvKind;

class ConvDimensionAdapter {
  explicit ConvDimensionAdapter(const HloInstruction& conv, ConvKind conv_kind,
                                ConvolutionDimensionNumbers dums)
      : conv_(conv), conv_kind_(conv_kind), dums_(dums) {}

 public:
  const HloInstruction& conv_;
  ConvKind conv_kind_;

  static absl::StatusOr<std::optional<ConvDimensionAdapter>> Create(
      const HloFusionInstruction& fusion, const HloComputation& computation) {
    const HloInstruction* maybe_conv = hlo_query::GetFirstInstructionWithOpcode(
        computation, HloOpcode::kConvolution);
    if (maybe_conv == nullptr) {
      VLOG(3) << "Not a Conv fusion.";
      return std::nullopt;
    }
    ConvKind conv_kind =
        DynCast<HloConvolutionInstruction>(maybe_conv)->conv_kind();

    ConvolutionDimensionNumbers dnums_for_layout =
        RestoreDimNumber(DynCast<HloConvolutionInstruction>(maybe_conv));

    // make sure input/kernel/output has the same layout
    TF_RET_CHECK(dnums_for_layout.input_batch_dimension() ==
                     dnums_for_layout.kernel_output_feature_dimension() &&
                 dnums_for_layout.kernel_output_feature_dimension() ==
                     dnums_for_layout.output_batch_dimension());
    TF_RET_CHECK(dnums_for_layout.input_feature_dimension() ==
                     dnums_for_layout.kernel_input_feature_dimension() &&
                 dnums_for_layout.kernel_input_feature_dimension() ==
                     dnums_for_layout.output_feature_dimension());
    for (auto i = 0; i < dnums_for_layout.input_spatial_dimensions_size();
         ++i) {
      TF_RET_CHECK(dnums_for_layout.input_spatial_dimensions(i) ==
                       dnums_for_layout.kernel_spatial_dimensions(i) &&
                   dnums_for_layout.kernel_spatial_dimensions(i) ==
                       dnums_for_layout.output_spatial_dimensions(i));
    }
    return ConvDimensionAdapter{*maybe_conv, conv_kind, dnums_for_layout};
  }

  std::optional<Result> DimensionsAndStrides(const HloInstruction& hlo) {
    if (ShapeUtil::IsScalar(hlo.shape())) {
      Result result;
      result.sizes =
          std::vector<int64_t>(dums_.input_spatial_dimensions_size() + 2, 1);
      result.strides =
          std::vector<int64_t>(dums_.input_spatial_dimensions_size() + 2, 1);
      return result;
    }
    // Placeholder FP32 data type here, it is not used
    auto desc = se::dnn::TensorDescriptor::For(
        se::dnn::DataType::kFloat, hlo.shape().dimensions(),
        hlo.shape().layout().minor_to_major());
    // Logical layout and physical layout should be the same after layout
    // assignment.
    std::vector<int64_t> logical_dims = desc.dimensions();
    std::vector<int64_t> logical_strides = desc.GetLogicalStrides();
    // We shouldn't need to know if this hlo is LHS, RHS or Output,
    // they should have same layout after layout assignment. Use input dums
    // here.
    Result result;
    result.sizes.push_back(logical_dims[dums_.input_batch_dimension()]);
    result.sizes.push_back(logical_dims[dums_.input_feature_dimension()]);
    result.strides.push_back(logical_strides[dums_.input_batch_dimension()]);
    result.strides.push_back(logical_strides[dums_.input_feature_dimension()]);
    for (auto i = 0; i < dums_.input_spatial_dimensions_size(); ++i) {
      result.sizes.push_back(logical_dims[dums_.input_spatial_dimensions(i)]);
      result.strides.push_back(
          logical_strides[dums_.input_spatial_dimensions(i)]);
    }
    return result;
  }

 private:
  ConvolutionDimensionNumbers dums_;
};

template <PrimitiveType XlaT, typename T>
std::shared_ptr<graph::Tensor_attributes> LiteralToCudnnTensor(
    const HloInstruction& hlo, graph::Graph& graph) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<XlaT>::type;
  return graph.tensor(T(hlo.literal().GetFirstElement<NativeT>()));
}

std::optional<std::shared_ptr<graph::Tensor_attributes>>
HandleConstantHloToCudnnGraph(const HloInstruction& hlo, graph::Graph& graph) {
  CHECK(hlo.IsConstant()) << "HLO is not a constant: " << hlo.ToShortString();
  if (!ShapeUtil::IsScalar(hlo.shape())) {
    VLOG(3) << "Currently only support fusing scalar in the graph";
    return std::nullopt;
  }
  PrimitiveType constant_type = hlo.shape().element_type();
  switch (constant_type) {
    case F16:
      return LiteralToCudnnTensor<F16, __half>(hlo, graph);
    case BF16:
      return LiteralToCudnnTensor<BF16, __nv_bfloat16>(hlo, graph);
    case F32:
      return LiteralToCudnnTensor<F32, float>(hlo, graph);
    case S32:
      return LiteralToCudnnTensor<S32, int>(hlo, graph);
    default:
      VLOG(3) << "Unsupported constant type: "
              << PrimitiveType_Name(constant_type);
      return std::nullopt;
  }
}

std::optional<std::shared_ptr<graph::Tensor_attributes>>
HandleClampToCudnnGraph(
    const HloInstruction& hlo, graph::Graph& graph,
    absl::flat_hash_map<const HloInstruction*,
                        std::shared_ptr<graph::Tensor_attributes>>
        hlo_to_cudnn,
    fe::DataType_t compute_dtype) {
  CHECK(hlo.opcode() == HloOpcode::kClamp)
      << "HLO is not a clamp: " << hlo.ToShortString();
  CHECK(hlo.operands().size() == 3)
      << "Clamp requires to have 3 operands: " << hlo.ToShortString();
  // clamp = max(lower, min(value, upper));
  const auto min_attrs = graph::Pointwise_attributes()
                             .set_mode(fe::PointwiseMode_t::MIN)
                             .set_compute_data_type(compute_dtype);
  std::shared_ptr<graph::Tensor_attributes> min_tensor = graph.pointwise(
      hlo_to_cudnn[hlo.operand(1)], hlo_to_cudnn[hlo.operand(2)], min_attrs);
  const std::optional<fe::DataType_t> data_type =
      ToCudnnDataType(hlo.shape().element_type());
  if (!data_type.has_value()) {
    VLOG(3) << "Unimplemented data type: "
            << PrimitiveType_Name(hlo.shape().element_type());
    return std::nullopt;
  }
  min_tensor->set_data_type(*data_type).set_name(std::string(hlo.name()));
  const auto max_attrs = graph::Pointwise_attributes()
                             .set_mode(fe::PointwiseMode_t::MAX)
                             .set_compute_data_type(compute_dtype);
  return graph.pointwise(min_tensor, hlo_to_cudnn[hlo.operand(0)], max_attrs);
}

std::optional<std::shared_ptr<graph::Tensor_attributes>>
HandleExpMinusOneToCudnnGraph(
    const HloInstruction& hlo, graph::Graph& graph,
    absl::flat_hash_map<const HloInstruction*,
                        std::shared_ptr<graph::Tensor_attributes>>
        hlo_to_cudnn,
    fe::DataType_t compute_dtype) {
  CHECK(hlo.opcode() == HloOpcode::kExpm1)
      << "HLO is not a Exp-minus-one: " << hlo.ToShortString();
  CHECK(hlo.operands().size() == 1)
      << "Exp-minus-one requires to have 1 operand: " << hlo.ToShortString();
  // exp-minus-one = exp(value) - 1;
  const auto exp_attrs = graph::Pointwise_attributes()
                             .set_mode(fe::PointwiseMode_t::EXP)
                             .set_compute_data_type(compute_dtype);
  std::shared_ptr<graph::Tensor_attributes> exp_tensor =
      graph.pointwise(hlo_to_cudnn[hlo.operand(0)], exp_attrs);
  const std::optional<fe::DataType_t> data_type =
      ToCudnnDataType(hlo.shape().element_type());
  if (!data_type.has_value()) {
    VLOG(3) << "Unimplemented data type: "
            << PrimitiveType_Name(hlo.shape().element_type());
    return std::nullopt;
  }
  exp_tensor->set_data_type(*data_type).set_name(std::string(hlo.name()));
  const auto minus_attrs = graph::Pointwise_attributes()
                               .set_mode(fe::PointwiseMode_t::SUB)
                               .set_compute_data_type(compute_dtype);
  return graph.pointwise(exp_tensor, graph.tensor(1), minus_attrs);
}

// Traverses fusion computations and creates cuDNN graphs out of them.
absl::StatusOr<std::optional<se::gpu::CudnnGraph>> HloFusionToCuDnnGraph(
    const HloFusionInstruction& fusion) {
  const HloComputation& computation = *fusion.fused_instructions_computation();
  VLOG(5) << fusion.ToString();
  VLOG(5) << computation.ToString();
  graph::Graph graph;
  // Intermediate data type is needed for `block_scale_dequantize` graph nodes.
  graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT);

  std::vector<HloInstruction*> instructions =
      computation.MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*,
                      std::shared_ptr<graph::Tensor_attributes>>
      hlo_to_cudnn;
  TF_ASSIGN_OR_RETURN(std::optional<GemmDimensionAdapter> gemm_adapter,
                      GemmDimensionAdapter::Create(computation));
  TF_ASSIGN_OR_RETURN(std::optional<ConvDimensionAdapter> conv_adapter,
                      ConvDimensionAdapter::Create(fusion, computation));
  if (!gemm_adapter.has_value() && !conv_adapter.has_value()) {
    VLOG(3) << "No dot or conv found inside cudnn fusion.";
    return std::nullopt;
  }

  auto add_parameter = [&](const HloInstruction& parameter,
                           const Result& dims) {
    const std::optional<fe::DataType_t> data_type =
        ToCudnnDataType(parameter.shape().element_type());
    if (!data_type.has_value()) {
      VLOG(3) << "Unsupported data type.";
      return false;
    }
    hlo_to_cudnn[&parameter] = graph.tensor(
        graph::Tensor_attributes()
            .set_dim(dims.sizes)
            .set_stride(dims.strides)
            .set_data_type(*data_type)
            .set_name(std::string(parameter.name()))
            .set_uid(se::gpu::CuDnnTensorUID(parameter.parameter_number())));
    if (dims.slices.has_value()) {
      hlo_to_cudnn[&parameter] = graph.slice(
          hlo_to_cudnn[&parameter],
          graph::Slice_attributes().set_slices(dims.slices.value()));
    }
    return true;
  };

  if (conv_adapter.has_value()) {
    for (const HloInstruction* parameter :
         computation.parameter_instructions()) {
      // for now, we assume all parameters have same layout even if they are not
      // inputs to conv, for example, bias add after conv.
      const std::optional<Result> dims =
          conv_adapter->DimensionsAndStrides(*parameter);
      VLOG(3) << "parameter: " << parameter->ToString() << "\n";
      if (!dims.has_value()) {
        VLOG(3) << "Unsupported dimensions.";
        return std::nullopt;
      }
      if (!add_parameter(*parameter, *dims)) {
        return std::nullopt;
      }
    }
  } else {
    // dot and scale dot
    for (const TritonFusionAnalysis::Scope scope :
         {TritonFusionAnalysis::Scope::LHS,
          TritonFusionAnalysis::Scope::LHS_SCALE,
          TritonFusionAnalysis::Scope::RHS,
          TritonFusionAnalysis::Scope::RHS_SCALE,
          TritonFusionAnalysis::Scope::OUTPUT}) {
      if (!gemm_adapter->analysis_.is_scaled_dot() &&
          (scope == TritonFusionAnalysis::Scope::LHS_SCALE ||
           scope == TritonFusionAnalysis::Scope::RHS_SCALE)) {
        continue;
      }
      for (const HloInstruction* parameter :
           gemm_adapter->analysis_.ScopeParameters(scope)) {
        const std::optional<Result> dims =
            gemm_adapter->DimensionsAndStrides(*parameter, scope);
        VLOG(3) << "parameter: " << parameter->ToString() << "\n";
        if (!dims.has_value()) {
          VLOG(3) << "Unsupported dimensions.";
          return std::nullopt;
        }
        if (!add_parameter(*parameter, *dims)) {
          return std::nullopt;
        }
      }
    }
  }

  for (const HloInstruction* hlo : instructions) {
    VLOG(5) << hlo->ToShortString();
    auto operand = [&hlo_to_cudnn, &hlo](int i) {
      return hlo_to_cudnn[hlo->operand(i)];
    };

    if (HloPredicateIsOp<HloOpcode::kConvert>(hlo) && hlo->user_count() == 1 &&
        HloPredicateIsOp<HloOpcode::kConvolution>(hlo->users()[0])) {
      // consume converts of inputs to conv, conv can do fp32 = conv(fp8, fp8)
      // and int32 = conv(int8, int8)
      hlo_to_cudnn[hlo] = operand(0);
      continue;
    } else if (HloPredicateIsOp<HloOpcode::kParameter>(hlo)) {
      CHECK(hlo_to_cudnn.contains(hlo));
      continue;
    } else if (HloPredicateIsOp<HloOpcode::kCustomCall>(hlo)) {
      if (hlo->user_count() != 1 ||
          !IsWorkspaceAllocationRoot(*hlo->users()[0])) {
        VLOG(3) << "Custom calls are only expected to be used for workspace "
                   "allocation.";
        return std::nullopt;
      }
      continue;
    } else if (HloPredicateIsOp<HloOpcode::kTuple>(hlo)) {
      if (!IsWorkspaceAllocationRoot(*hlo) && !IsAmaxRoot(*hlo)) {
        VLOG(3) << "Tuples are only expected at outputs for workspace "
                   "allocation.";
        return std::nullopt;
      }
      continue;
    } else if (HloPredicateIsOp<HloOpcode::kConstant>(hlo)) {
      if (const auto const_tensor = HandleConstantHloToCudnnGraph(*hlo, graph);
          const_tensor.has_value()) {
        hlo_to_cudnn[hlo] = const_tensor.value();
      } else {
        return std::nullopt;
      }
    } else if (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kBitcast,
                                HloOpcode::kTranspose, HloOpcode::kCopy,
                                HloOpcode::kBroadcast, HloOpcode::kSlice>(
                   hlo)) {
      // All these are accounted for separately as transformations of strides.
      hlo_to_cudnn[hlo] = operand(0);
    } else if (hlo->IsElementwise()) {
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }
      if (HloPredicateIsOp<HloOpcode::kClamp>(hlo)) {
        const auto clamp = HandleClampToCudnnGraph(*hlo, graph, hlo_to_cudnn,
                                                   compute_dtype.value());
        if (!clamp.has_value()) {
          return std::nullopt;
        }
        hlo_to_cudnn[hlo] = clamp.value();
      } else if (HloPredicateIsOp<HloOpcode::kExpm1>(hlo)) {
        const auto expm1 = HandleExpMinusOneToCudnnGraph(
            *hlo, graph, hlo_to_cudnn, compute_dtype.value());
        if (!expm1.has_value()) {
          return std::nullopt;
        }
        hlo_to_cudnn[hlo] = expm1.value();
      } else {
        const auto mode = GetElementwiseMode(*hlo);
        if (!mode.has_value()) {
          VLOG(3) << "Unsupported elementwise operation.";
          return std::nullopt;
        }
        const auto attrs = graph::Pointwise_attributes()
                               .set_mode(mode.value())
                               .set_compute_data_type(compute_dtype.value());
        if (hlo->operand_count() == 1) {
          hlo_to_cudnn[hlo] = graph.pointwise(operand(0), attrs);
          // Sets the dimensions for unary ops whose operands are broadcast
          // for cuDNN to infer its inputs' shapes. constant has dimension [1]
          // while cuDNN requires constant to have dimension [1,1,1]. Not
          // setting output of the unary shapes results in the rejection of
          // the cuDNN graph.
          if (hlo->operand(0)->opcode() == HloOpcode::kBroadcast) {
            const std::optional<TritonFusionAnalysis::Scope> scope =
                gemm_adapter->analysis_.QueryInstructionScope(*hlo);
            if (!scope.has_value()) {
              LOG(FATAL) << "No scope for instruction: "
                         << hlo->ToShortString();
            }
            const std::optional<Result> dims =
                gemm_adapter->DimensionsAndStrides(*hlo, *scope);
            if (!dims.has_value()) {
              VLOG(3) << "Unsupported hlo for querying dimensions: "
                      << hlo->ToShortString();
            } else {
              hlo_to_cudnn[hlo]->set_dim(dims->sizes);
            }
          }
        } else if (hlo->operand_count() == 2) {
          hlo_to_cudnn[hlo] = graph.pointwise(operand(0), operand(1), attrs);
        } else if (hlo->operand_count() == 3) {
          if (HloPredicateIsNotOp<HloOpcode::kSelect>(hlo)) {
            VLOG(3) << "Unexpected ternary operation: " << hlo->ToString();
            return std::nullopt;
          }
          // Operand order for select differs between HLO and cuDNN.
          hlo_to_cudnn[hlo] =
              graph.pointwise(operand(1), operand(2), operand(0), attrs);
        } else {
          VLOG(3) << "Unimplemented elementwise operation.";
          return std::nullopt;
        }
      }
    } else if (HloPredicateIsOp<HloOpcode::kDot>(hlo)) {
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }
      hlo_to_cudnn[hlo] =
          graph.matmul(operand(0), operand(1),
                       graph::Matmul_attributes().set_compute_data_type(
                           compute_dtype.value()));
    } else if (HloPredicateIsOp<HloOpcode::kScaledDot>(hlo)) {
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }
      std::array<std::shared_ptr<graph::Tensor_attributes>, 2> dot_operands;
      for (int i = 0; i < 2; ++i) {
        const Shape& scale_shape = hlo->operand(i + 2)->shape();
        int block_size = scale_shape.element_type() == F8E8M0FNU
                             ? BlockScalingRewriter::kBlockSizeMXFP8
                             : BlockScalingRewriter::kBlockSizeNVFP4;
        auto scale = operand(i + 2);
        scale->set_reordering_type(fe::TensorReordering_t::F8_128x4);
        auto dq_attrs = graph::Block_scale_dequantize_attributes()
                            .set_block_size(block_size)
                            .set_compute_data_type(*compute_dtype);
        dot_operands[i] =
            graph.block_scale_dequantize(operand(i), scale, dq_attrs);
        dot_operands[i]->set_name(
            absl::StrCat(hlo->name(), i == 0 ? "_lhs" : "_rhs", "_dq"));
      }
      hlo_to_cudnn[hlo] = graph.matmul(
          dot_operands[0], dot_operands[1],
          graph::Matmul_attributes().set_compute_data_type(*compute_dtype));
    } else if (HloPredicateIsOp<HloOpcode::kConvolution>(hlo)) {
      // translate conv windows to cudnn conv attr
      std::optional<Window> window_opt =
          RestoreWindow(DynCast<HloConvolutionInstruction>(hlo));
      CHECK(window_opt.has_value());
      Window window = window_opt.value();
      std::vector<int64_t> pre_padding, post_padding, stride, dilation;
      for (int64_t i = 0; i < window.dimensions_size(); ++i) {
        const auto& dim = window.dimensions(i);
        pre_padding.push_back(dim.padding_low());
        post_padding.push_back(dim.padding_high());
        stride.push_back(dim.stride());
        dilation.push_back(dim.window_dilation());
      }
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }

      // lower to different conv based on conv_kind set in cudnn fusion backend
      // config
      auto set_conv_attr = [&](auto conv_attr) {
        return conv_attr.set_pre_padding(pre_padding)
            .set_post_padding(post_padding)
            .set_stride(stride)
            .set_dilation(dilation)
            .set_compute_data_type(compute_dtype.value());
      };
      if (conv_adapter->conv_kind_ == ConvKind::FPROP) {
        hlo_to_cudnn[hlo] =
            graph.conv_fprop(operand(0), operand(1),
                             set_conv_attr(graph::Conv_fprop_attributes()));
      } else if (conv_adapter->conv_kind_ == ConvKind::DGRAD) {
        hlo_to_cudnn[hlo] =
            graph.conv_dgrad(operand(0), operand(1),
                             set_conv_attr(graph::Conv_dgrad_attributes()));
      } else if (conv_adapter->conv_kind_ == ConvKind::WGRAD) {
        // cudnn frontend accepts operand in the order of dout, input, but xla
        // uses reverse order
        hlo_to_cudnn[hlo] =
            graph.conv_wgrad(operand(1), operand(0),
                             set_conv_attr(graph::Conv_wgrad_attributes()));
      } else {
        VLOG(3) << "Unimplemented conv type.";
        return std::nullopt;
      }
      // cuDNN requires output dims to be set for conv dgrad and wgrad, it is
      // not required for fprop but we do it anyway for simplicity
      const std::optional<Result> dims =
          conv_adapter->DimensionsAndStrides(*hlo);
      hlo_to_cudnn[hlo]->set_dim(dims->sizes);
    } else if (HloPredicateIsOp<HloOpcode::kReduce>(hlo)) {
      hlo_to_cudnn[hlo] = graph.reduction(
          operand(0), graph::Reduction_attributes()
                          .set_mode(fe::ReductionMode_t::AMAX)
                          .set_compute_data_type(fe::DataType_t::FLOAT));
    } else {
      VLOG(3) << "Unimplemented operation.";
      return std::nullopt;
    }
    if (hlo_to_cudnn[hlo] == nullptr) {
      VLOG(3) << "Creation of the operation failed.";
      return std::nullopt;
    }
    const std::optional<fe::DataType_t> data_type =
        ToCudnnDataType(hlo->shape().element_type());
    if (!data_type.has_value()) {
      VLOG(3) << "Unimplemented data type: "
              << PrimitiveType_Name(hlo->shape().element_type());
      return std::nullopt;
    }
    hlo_to_cudnn[hlo]
        ->set_data_type(data_type.value())
        .set_name(std::string(hlo->name()));
  }

  std::vector<HloInstruction*> outputs;
  if (instructions.back()->shape().IsTuple()) {
    for (auto operand : instructions.back()->operands()) {
      if (!operand->IsCustomCall(kWorkspaceAllocationCustomCallTarget)) {
        outputs.push_back(operand);
      }
    }
  } else {
    outputs.push_back(instructions.back());
  }

  for (int i = 0; i < outputs.size(); ++i) {
    HloInstruction* output = outputs[i];
    const std::optional<Result> dims =
        conv_adapter.has_value()
            ? conv_adapter->DimensionsAndStrides(*output)
            : gemm_adapter->DimensionsAndStrides(
                  *output, TritonFusionAnalysis::Scope::OUTPUT);
    if (!dims.has_value()) {
      VLOG(3) << "Unsupported dimensions.";
      return std::nullopt;
    }
    hlo_to_cudnn[output]
        ->set_output(true)
        .set_dim(dims->sizes)
        .set_stride(dims->strides)
        .set_uid(se::gpu::CuDnnTensorUID(fusion.operand_count() + i));
  }

  if (!fusion.GetModule()->config().debug_options().xla_dump_to().empty()) {
    json dump;
    graph.serialize(dump);
    DumpToFileInDirOrStdout(
        /*module=*/*fusion.GetModule(),
        /*file_prefix=*/"",
        /*file_suffix=*/
        absl::StrCat("cudnn_fusion_", fusion.name(), ".json"),
        /*contents=*/dump.dump(1));
  }

  return se::gpu::CudnnGraph(std::move(graph));
}

// Creates a cuDNN graph, queries cuDNN whether it is supported.
absl::StatusOr<se::gpu::CudnnGraph> PrepareGraph(
    se::dnn::DnnSupport& dnn_support, const HloFusionInstruction& hlo) {
  TF_ASSIGN_OR_RETURN(std::optional<se::gpu::CudnnGraph> graph,
                      HloFusionToCuDnnGraph(hlo));
  if (!graph.has_value()) {
    return absl::InternalError("Construction of cuDNN graph failed.");
  }
  TF_RETURN_IF_ERROR(graph->Prepare(
      dnn_support, se::EngineOptions{
                       RequireDeterminism(hlo.GetModule()->config()),
                       /*allow_tf32=*/true, /*require_command_buffer=*/false}));
  return *graph;
}

absl::StatusOr<HloInstruction*> AddWorkspace(HloInstruction& fusion,
                                             const int64_t workspace_size) {
  HloComputation* computation = fusion.fused_instructions_computation();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          ShapeUtil::MakeShape(S8, {workspace_size}), {},
          kWorkspaceAllocationCustomCallTarget));
  HloInstruction* output_tuple;
  bool is_tuple_output =
      computation->root_instruction()->opcode() == HloOpcode::kTuple;
  if (is_tuple_output) {
    std::vector<HloInstruction*> operands;
    operands.insert(operands.begin(),
                    computation->root_instruction()->operands().begin(),
                    computation->root_instruction()->operands().end());
    operands.push_back(custom_call);
    output_tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(operands));
    TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
        computation->root_instruction(), output_tuple));
  } else {
    output_tuple = computation->AddInstruction(HloInstruction::CreateTuple(
        {computation->root_instruction(), custom_call}));
  }
  computation->set_root_instruction(output_tuple, true);
  HloInstruction* new_fusion = fusion.parent()->AddInstruction(
      fusion.CloneWithNewShape(output_tuple->shape()));
  TF_RETURN_IF_ERROR(new_fusion->CopyAllControlDepsFrom(&fusion));
  TF_RETURN_IF_ERROR(fusion.DropAllControlDeps());
  if (is_tuple_output) {
    TF_RETURN_IF_ERROR(fusion.parent()->ReplaceInstructionWithDifferentShape(
        &fusion, new_fusion));
  } else {
    TF_RETURN_IF_ERROR(
        fusion.ReplaceAllUsesWith(fusion.parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(new_fusion, 0))));
    TF_RETURN_IF_ERROR(fusion.parent()->RemoveInstruction(&fusion));
  }
  return new_fusion;
}

class CuDnnFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CuDnnFusionVisitor(se::dnn::DnnSupport& dnn_support,
                              BinaryMap& compilation_results)
      : dnn_support_(dnn_support), compilation_results_(compilation_results) {}

  absl::Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    const FusionBackendConfig& fusion_backend_config =
        gpu_config.fusion_backend_config();
    if (fusion_backend_config.kind() != kCuDnnFusionKind) {
      return absl::OkStatus();
    }
    VLOG(4) << "Processing " << hlo->ToString();

    auto compile_graph = [&]() -> absl::StatusOr<se::gpu::CudnnGraph> {
      TF_ASSIGN_OR_RETURN(
          se::gpu::CudnnGraph graph,
          PrepareGraph(dnn_support_, *DynCast<HloFusionInstruction>(hlo)));

      if (fusion_backend_config.has_cudnn_fusion_config() &&
          fusion_backend_config.cudnn_fusion_config().plan_id() >= 0) {
        const int64_t plan_id =
            fusion_backend_config.cudnn_fusion_config().plan_id();
        VLOG(4) << "Plan ID: " << plan_id;
        // Build single plan with given ID.
        if (plan_id >= graph.Graph().get_execution_plan_count()) {
          return absl::InternalError("cuDNN graph plan does not exist.");
        }
        TF_RETURN_IF_ERROR(graph.Build(dnn_support_, plan_id));
      } else {
        // Build plans one by one till first successful when no plan_id was
        // provided.
        int64_t plan_id = 0;
        for (; plan_id < graph.Graph().get_execution_plan_count(); ++plan_id) {
          VLOG(7) << "Trying plan ID " << plan_id;
          if (graph.Build(dnn_support_, plan_id).ok()) {
            VLOG(7) << "Successfully built plan ID " << plan_id;
            break;
          }
        }
        if (plan_id == graph.Graph().get_execution_plan_count()) {
          return absl::InternalError("No cuDNN plans can be built.");
        }
        CuDnnFusionConfig* cudnn_config =
            gpu_config.mutable_fusion_backend_config()
                ->mutable_cudnn_fusion_config();
        cudnn_config->set_plan_id(plan_id);
        TF_RETURN_IF_ERROR(hlo->set_backend_config(gpu_config));
      }
      return graph;
    };

    auto serialize_graph =
        [](const se::gpu::CudnnGraph& graph) -> absl::StatusOr<std::string> {
      std::vector<uint8_t> serialized_graph;
      RETURN_IF_CUDNN_FRONTEND_ERROR(graph.Graph().serialize(serialized_graph));
      return std::string(reinterpret_cast<char*>(serialized_graph.data()),
                         serialized_graph.size());
    };

    if (IsWorkspaceAllocationRoot(*hlo->fused_expression_root())) {
      // The graph already has a workspace.
      const std::string fingerprint = emitters::GetComputationFingerprint(
          hlo->fused_instructions_computation(), {});
      if (auto it = compilation_results_.find(fingerprint);
          it == compilation_results_.cend()) {
        TF_ASSIGN_OR_RETURN(const se::gpu::CudnnGraph graph, compile_graph());
        TF_ASSIGN_OR_RETURN(const std::string serialized,
                            serialize_graph(graph));
        compilation_results_.insert(it, {fingerprint, serialized});
      }
      return absl::OkStatus();
    }

    auto add_workspace = [&](const int64_t workspace_size) {
      if (workspace_size > 0) {
        TF_ASSIGN_OR_RETURN(hlo, AddWorkspace(*hlo, workspace_size));
        SetVisited(*hlo);
      }
      return absl::OkStatus();
    };

    const std::string fingerprint_without_workspace =
        emitters::GetComputationFingerprint(
            hlo->fused_instructions_computation(), {});

    auto workspace_size_it =
        workspace_sizes_.find(fingerprint_without_workspace);
    if (workspace_size_it == workspace_sizes_.cend()) {
      TF_ASSIGN_OR_RETURN(const se::gpu::CudnnGraph graph, compile_graph());
      const int64_t workspace_size = graph.Graph().get_workspace_size();
      workspace_sizes_.insert(workspace_size_it,
                              {fingerprint_without_workspace, workspace_size});
      TF_RETURN_IF_ERROR(add_workspace(workspace_size));
      TF_ASSIGN_OR_RETURN(const std::string serialized, serialize_graph(graph));
      compilation_results_[emitters::GetComputationFingerprint(
          hlo->fused_instructions_computation(), {})] = serialized;
    } else {
      VLOG(4) << "Cache hit.";
      TF_RETURN_IF_ERROR(add_workspace(workspace_size_it->second));
    }

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  se::dnn::DnnSupport& dnn_support_;
  // <HLO computation fingerprint, serialized compiled cuDNN graph>.
  BinaryMap& compilation_results_;
  absl::flat_hash_map<std::string, int64_t> workspace_sizes_;
};

}  // namespace

absl::StatusOr<bool> CuDnnFusionCompiler::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("cuDNN fusion compiler");
  return CuDnnFusionVisitor(dnn_support_, compilation_results_)
      .RunOnModule(module, execution_threads);
}

int CuDnnFusionCompiler::GetAvailablePlanCount(
    se::StreamExecutor& stream_exec, const HloFusionInstruction& hlo) {
  auto graph = PrepareGraph(*stream_exec.AsDnn(), hlo);
  if (!graph.ok()) {
    VLOG(1) << "Failed to prepare graph: " << graph.status();
    return 0;
  }
  return std::min(
      static_cast<int32_t>(graph->Graph().get_execution_plan_count()),
      hlo.GetModule()->config().debug_options().xla_gpu_cudnn_gemm_max_plans());
}

}  // namespace gpu
}  // namespace xla
