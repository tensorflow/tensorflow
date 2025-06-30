/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cudnn_sdpa_score_mod.h"

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/dnn.h"

namespace stream_executor {
namespace gpu {

ScoreModFunc::ScoreModFunc(const xla::HloComputation* fwd_comp,
                           const xla::HloComputation* bwd_comp)
    : fwd_comp_(fwd_comp), bwd_comp_(bwd_comp) {}

std::optional<cudnn_frontend::PointwiseMode_t> GetElementwiseMode(
    const xla::HloInstruction& instruction) {
  const xla::HloOpcode opcode = instruction.opcode();
  using m = cudnn_frontend::PointwiseMode_t;
  switch (opcode) {
    case xla::HloOpcode::kAbs:
      return m::ABS;
    case xla::HloOpcode::kAdd:
      return m::ADD;
    case xla::HloOpcode::kCeil:
      return m::CEIL;
    case xla::HloOpcode::kCompare:
      switch (instruction.comparison_direction()) {
        case xla::Comparison::Direction::kEq:
          return m::CMP_EQ;
        case xla::Comparison::Direction::kNe:
          return m::CMP_NEQ;
        case xla::Comparison::Direction::kGe:
          return m::CMP_GE;
        case xla::Comparison::Direction::kGt:
          return m::CMP_GT;
        case xla::Comparison::Direction::kLe:
          return m::CMP_LE;
        case xla::Comparison::Direction::kLt:
          return m::CMP_LT;
      }
      break;
    case xla::HloOpcode::kConvert:
      return m::IDENTITY;
    case xla::HloOpcode::kCos:
      return m::COS;
    case xla::HloOpcode::kDivide:
      return m::DIV;
    case xla::HloOpcode::kExp:
      return m::EXP;
    case xla::HloOpcode::kFloor:
      return m::FLOOR;
    case xla::HloOpcode::kLog:
      return m::LOG;
    case xla::HloOpcode::kMaximum:
      return m::MAX;
    case xla::HloOpcode::kMinimum:
      return m::MIN;
    case xla::HloOpcode::kMultiply:
      return m::MUL;
    case xla::HloOpcode::kNegate:
      return m::NEG;
    case xla::HloOpcode::kPower:
      return m::POW;
    case xla::HloOpcode::kRsqrt:
      return m::RSQRT;
#if CUDNN_VERSION >= 90100
    case xla::HloOpcode::kSelect:
      return m::BINARY_SELECT;
#endif  // CUDNN_VERSION
    case xla::HloOpcode::kSin:
      return m::SIN;
    case xla::HloOpcode::kSqrt:
      return m::SQRT;
    case xla::HloOpcode::kSubtract:
      return m::SUB;
    case xla::HloOpcode::kTan:
      return m::TAN;
    case xla::HloOpcode::kTanh:
      return m::TANH_FWD;
    case xla::HloOpcode::kAnd:
      return m::LOGICAL_AND;
    case xla::HloOpcode::kOr:
      return m::LOGICAL_OR;
    default:
      return std::nullopt;
  }
}

cudnn_frontend::DataType_t GetComputeDataType(const xla::PrimitiveType type) {
  cudnn_frontend::DataType_t compute_dtype = cudnn_frontend::DataType_t::FLOAT;
  if (xla::primitive_util::IsIntegralType(type)) {
    compute_dtype = cudnn_frontend::DataType_t::INT32;
  } else if (type == xla::PrimitiveType::PRED) {
    compute_dtype = cudnn_frontend::DataType_t::BOOLEAN;
  }
  return compute_dtype;
}

cudnn_frontend::DataType_t ToCudnnFrontendDataType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return cudnn_frontend::DataType_t::FLOAT;
    case dnn::DataType::kDouble:
      return cudnn_frontend::DataType_t::DOUBLE;
    case dnn::DataType::kHalf:
      return cudnn_frontend::DataType_t::HALF;
    case dnn::DataType::kInt8:
      return cudnn_frontend::DataType_t::INT8;
    case dnn::DataType::kInt32:
      return cudnn_frontend::DataType_t::INT32;
    case dnn::DataType::kInt64:
      return cudnn_frontend::DataType_t::INT64;
    case dnn::DataType::kBF16:
      return cudnn_frontend::DataType_t::BFLOAT16;
    case dnn::DataType::kF8E4M3FN:
      return cudnn_frontend::DataType_t::FP8_E4M3;
    case dnn::DataType::kF8E5M2:
      return cudnn_frontend::DataType_t::FP8_E5M2;
#if CUDNN_VERSION >= 90700
    case dnn::DataType::kF4E2M1FN:
      return cudnn_frontend::DataType_t::FP4_E2M1;
    case dnn::DataType::kF8E8M0FNU:
      return cudnn_frontend::DataType_t::FP8_E8M0;
#endif
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

template <xla::PrimitiveType XlaT, typename T>
Tensor LiteralToCudnnTensor(const xla::HloInstruction* hlo,
                            cudnn_frontend::graph::Graph& graph) {
  using NativeT =
      typename xla::primitive_util::PrimitiveTypeToNative<XlaT>::type;
  return graph.tensor(T(hlo->literal().GetFirstElement<NativeT>()));
}

absl::Status ScoreModFunc::UpdateCudnnMap(cudnn_frontend::graph::Graph& graph,
                                          UidGenerator next_uid) {
  TF_RETURN_IF_ERROR(UpdateHloParameterToCudnnMap(graph, fwd_hlo_to_cudnn_,
                                                  fwd_comp_, next_uid));
  TF_RETURN_IF_ERROR(
      UpdateHloConstantToCudnnMap(graph, fwd_hlo_to_cudnn_, fwd_comp_));
  if (bwd_comp_) {
    TF_RETURN_IF_ERROR(
        UpdateHloConstantToCudnnMap(graph, bwd_hlo_to_cudnn_, bwd_comp_));
  }
  return absl::OkStatus();
}

absl::Status ScoreModFunc::UpdateHloParameterToCudnnMap(
    cudnn_frontend::graph::Graph& graph,
    absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
    const xla::HloComputation* computation, UidGenerator next_uid) {
  for (int i = 1; i < computation->num_parameters(); i++) {
    auto parameter = computation->parameter_instruction(i);
    TF_ASSIGN_OR_RETURN(const dnn::DataType type,
                        xla::gpu::GetDNNDataTypeFromPrimitiveType(
                            parameter->shape().element_type()));
    auto desc = dnn::TensorDescriptor::For(
        type, parameter->shape().dimensions(),
        parameter->shape().layout().minor_to_major());
    auto dims = desc.dimensions();
    auto strides = desc.GetLogicalStrides();
    auto rank = dims.size();
    for (int i = 0; i < 4 - rank; i++) {
      // Pad dims and strides to rank 4
      dims.push_back(1);
      strides.push_back(1);
    }
    hlo_to_cudnn[parameter] =
        graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                         .set_dim(dims)
                         .set_stride(strides)
                         .set_data_type(ToCudnnFrontendDataType(desc.type()))
                         .set_name(absl::StrCat("score_mod_input_", i))
                         .set_uid(next_uid()));
  }
  return absl::OkStatus();
}

absl::Status ScoreModFunc::UpdateHloConstantToCudnnMap(
    cudnn_frontend::graph::Graph& graph,
    absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
    const xla::HloComputation* computation) {
  for (auto instr : computation->MakeInstructionPostOrder()) {
    if (xla::HloPredicateIsOp<xla::HloOpcode::kConstant>(instr)) {
      if (!xla::ShapeUtil::IsScalar(instr->shape())) {
        return absl::InternalError("Only support scalar.");
      }
      xla::PrimitiveType constant_type = instr->shape().element_type();
      switch (constant_type) {
        case xla::F16:
          hlo_to_cudnn[instr] =
              LiteralToCudnnTensor<xla::F16, __half>(instr, graph);
          break;
        case xla::BF16:
          hlo_to_cudnn[instr] =
              LiteralToCudnnTensor<xla::BF16, __nv_bfloat16>(instr, graph);
          break;
        case xla::F32:
          hlo_to_cudnn[instr] =
              LiteralToCudnnTensor<xla::F32, float>(instr, graph);
          break;
        case xla::S32:
          hlo_to_cudnn[instr] =
              LiteralToCudnnTensor<xla::S32, int>(instr, graph);
          break;
        default:
          return absl::InternalError("Unsupported constant type.");
      }
    }
  }
  return absl::OkStatus();
}

Tensor ScoreModFunc::Forward(Graph graph, Tensor attention_score) {
  fwd_hlo_to_cudnn_[fwd_comp_->parameter_instruction(0)] = attention_score;
  // save fwd parameters tensor attributes
  for (int i = 0; i < fwd_comp_->num_parameters(); i++) {
    auto parameter = fwd_comp_->parameter_instruction(i);
    fwd_parameters_.push_back(fwd_hlo_to_cudnn_[parameter]);
  }
  return Compile(graph, fwd_hlo_to_cudnn_, fwd_comp_);
}

Tensor ScoreModFunc::Backward(Graph graph, Tensor grad) {
  bwd_hlo_to_cudnn_[bwd_comp_->parameter_instruction(0)] = grad;
  // add fwd parameters tensor attribute to map
  for (int i = 1; i < bwd_comp_->num_parameters(); i++) {
    auto parameter = bwd_comp_->parameter_instruction(i);
    bwd_hlo_to_cudnn_[parameter] = fwd_parameters_[i - 1];
  }
  return Compile(graph, bwd_hlo_to_cudnn_, bwd_comp_);
}

Tensor ScoreModFunc::Compile(
    Graph graph,
    absl::flat_hash_map<const xla::HloInstruction*, Tensor>& hlo_to_cudnn,
    const xla::HloComputation* computation) {
  VLOG(3) << "Compiling cudnn sdpa computation:\n"
          << computation->ToString() << "\n";
  std::vector<xla::HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  for (const xla::HloInstruction* hlo : instructions) {
    auto operand = [&hlo_to_cudnn, &hlo](int i) {
      CHECK(hlo_to_cudnn.count(hlo->operand(i)));
      return hlo_to_cudnn[hlo->operand(i)];
    };
    // check if operand i is user input to the flex graph
    // all parameters except param 0 should return true
    // param 0 is managed by cuDNN
    auto is_virtual = [&hlo_to_cudnn, &hlo](int i) {
      return hlo_to_cudnn[hlo->operand(i)]->get_is_virtual();
    };

    if (xla::HloPredicateIsOp<xla::HloOpcode::kConstant,
                              xla::HloOpcode::kParameter>(hlo)) {
      continue;
    }
    if (xla::HloPredicateIsOp<xla::HloOpcode::kBitcast,
                              xla::HloOpcode::kBroadcast>(hlo)) {
      hlo_to_cudnn[hlo] = operand(0);
    } else if (xla::HloPredicateIsOp<xla::HloOpcode::kIota>(hlo)) {
      auto iota = DynCast<xla::HloIotaInstruction>(hlo);
      const auto attrs =
          cudnn_frontend::graph::Pointwise_attributes()
              .set_mode(cudnn_frontend::PointwiseMode_t::GEN_INDEX)
              .set_compute_data_type(cudnn_frontend::DataType_t::INT32)
              .set_axis(iota->iota_dimension())
              .set_name(std::string(hlo->name()));
      auto attn_score = computation->parameter_instruction(0);
      hlo_to_cudnn[hlo] = graph->pointwise(hlo_to_cudnn[attn_score], attrs);
      hlo_to_cudnn[hlo]->set_data_type(cudnn_frontend::DataType_t::INT32);
    } else if (hlo->IsElementwise()) {
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      const auto mode = GetElementwiseMode(*hlo);
      if (!mode.has_value()) {
        LOG(FATAL) << "Unsupported elementwise operation: " << hlo->ToString()
                   << "\n";
      }
      auto attrs = cudnn_frontend::graph::Pointwise_attributes()
                       .set_mode(mode.value())
                       .set_compute_data_type(compute_dtype)
                       .set_name(std::string(hlo->name()));
      if (hlo->operand_count() == 1) {
        hlo_to_cudnn[hlo] = graph->pointwise(operand(0), attrs);
      } else if (hlo->operand_count() == 2) {
        if (!is_virtual(0) && !is_virtual(1)) {
          LOG(FATAL)
              << "cuDNN doesn't support both operands to pointwise op to "
              << "be non virtual for now.";
        }
        // make sure first operand is virtual
        // remove this once cuDNN supports this
        if (!is_virtual(0) && !HloOpcodeIsBinaryCommutative(hlo->opcode())) {
          std::cerr << hlo->ToString() << "\n";
          LOG(FATAL) << "first operand of cuDNN pointwise op is not virtual "
                        "and op is not commutative.";
        }
        auto o0 = is_virtual(0) ? operand(0) : operand(1);
        auto o1 = is_virtual(0) ? operand(1) : operand(0);
        if (hlo->opcode() == xla::HloOpcode::kAnd ||
            hlo->opcode() == xla::HloOpcode::kOr) {
          // Constraint of cudnn logical operations to only accept int32
          // inputs. Remove this once cudnn supports boolean inputs.
          if (o0->get_data_type() != cudnn_frontend::DataType_t::INT32 &&
              o1->get_data_type() != cudnn_frontend::DataType_t::INT32) {
            attrs.set_compute_data_type(cudnn_frontend::DataType_t::INT32);
            auto convert =
                cudnn_frontend::graph::Pointwise_attributes()
                    .set_mode(cudnn_frontend::PointwiseMode_t::IDENTITY)
                    .set_compute_data_type(cudnn_frontend::DataType_t::INT32);
            o0 = graph->pointwise(o0, convert);
            o1 = graph->pointwise(o1, convert);
            o0->set_data_type(cudnn_frontend::DataType_t::INT32);
            o1->set_data_type(cudnn_frontend::DataType_t::INT32);
          }
        }
        hlo_to_cudnn[hlo] = graph->pointwise(o0, o1, attrs);
      } else if (hlo->operand_count() == 3) {
        if (xla::HloPredicateIsNotOp<xla::HloOpcode::kSelect>(hlo)) {
          LOG(FATAL) << "Unimplemented elementwise operation:"
                     << hlo->ToString() << "\n";
        }
        // Operand order for select differs between HLO and cuDNN.
        hlo_to_cudnn[hlo] =
            graph->pointwise(operand(1), operand(2), operand(0), attrs);
      } else {
        LOG(FATAL) << "Unimplemented elementwise operation:" << hlo->ToString()
                   << "\n";
      }
      hlo_to_cudnn[hlo]->set_data_type(compute_dtype);
    } else {
      LOG(FATAL) << "Unimplemented operation:" << hlo->ToString() << "\n";
    }
  }

  // return last output
  CHECK(hlo_to_cudnn.contains(computation->root_instruction()));
  return hlo_to_cudnn[computation->root_instruction()];
}

}  // namespace gpu
}  // namespace stream_executor
