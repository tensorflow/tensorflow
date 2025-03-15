/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

bool IsConvCustomCall(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kCustomCall>(instr) &&
         (instr->custom_call_target() == kCudnnConvForwardCallTarget ||
          instr->custom_call_target() ==
              kCudnnConvBiasActivationForwardCallTarget);
}

bool IsConvDepthwise(const HloInstruction* instr) {
  int64_t feature_group_count = instr->feature_group_count();
  if (feature_group_count == 1) {
    return false;
  }

  const HloInstruction* input = instr->operand(0);
  int64_t input_feature_dimension =
      instr->convolution_dimension_numbers().input_feature_dimension();
  int64_t input_feature_count =
      input->shape().dimensions(input_feature_dimension);
  return input_feature_count == feature_group_count;
}

// We don't want to upgrade depthwise convolutions to ConvBiasActivation,
// because the fused CUDNN functions are slower for some of those.
bool IsNonDepthwiseConvCustomCall(const HloInstruction* instr) {
  return IsConvCustomCall(instr) && !IsConvDepthwise(instr);
}

bool IsROCm(se::GpuComputeCapability cc) {
  return std::holds_alternative<se::RocmComputeCapability>(cc);
}

// elu, relu6, and leaky-relu activations are supported in cudnn via the
// "runtime fusion" engine, which JIT compiles C++ code.  This can be slow to
// compile, so we guard it with a debug option.
//
// nvidia currently recommends that we enable this only on Ampere+, but we've
// tested on Turing (sm75) and it seems to work fine.
//
// Note that as of writing, xla_gpu_use_runtime_fusion is disabled by default
// due to apparent bugs in cudnn 8.9.0.  See debug_options_flags.cc for details.
bool ShouldUseCudnnRuntimeFusion(const DebugOptions& debug_opts,
                                 se::GpuComputeCapability cc) {
  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&cc);
  if (cuda_cc != nullptr)
    return debug_opts.xla_gpu_use_runtime_fusion() && cuda_cc->IsAtLeast(7, 5);
  else
    return true;
}

bool IsSuitableForCudnnRuntimeFusion(HloInstruction* conv) {
  // cudnn runtime fusion is pathologically slow on convs with side-inputs.
  // TODO(kaixih@nvidia): remove this check when cuDNN fixes it.
  if (conv->operands().size() > 3) {
    return false;
  }

  // cuDNN runtime funsion kernels require 32-bit aligned data access, which
  // means that the number of in/out channels must be divisible by 2 for fp16.
  // (We don't currently do runtime fusion for int8.)
  if (conv->operand(0)->shape().element_type() != F16) {
    return false;
  }
  const Shape& shape = conv->operand(1)->shape();
  int64_t num_input_features = shape.dimensions(
      conv->convolution_dimension_numbers().kernel_input_feature_dimension());
  int64_t num_output_features = shape.dimensions(
      conv->convolution_dimension_numbers().kernel_output_feature_dimension());
  if (num_input_features % 2 != 0 || num_output_features % 2 != 0) {
    return false;
  }

  return true;
}

// Can instr be converted to type `dst_ty` without losing any precision?  For
// our purposes, this is true if:
//
//  - instr already has type dst_ty, or
//  - instr is convert<wider type>(op_with_dst_ty), or
//  - instr is a constant which we can convert orig_ty -> dst_ty -> orig_ty and
//    get back exactly the original value, or
//  - instr is a broadcast, reshape, or transpose of one of the above.
bool IsLosslesslyConvertibleTo(const HloInstruction* instr,
                               PrimitiveType dst_ty) {
  if (instr->shape().element_type() == dst_ty) {
    return true;
  }

  if (Match(instr, m::Convert(m::Op().WithElementType(dst_ty)))) {
    // Check that the convert from dst_ty to instr->element_type() doesn't lose
    // precision.  Otherwise, this convert is not lossless.
    return primitive_util::CastPreservesValues(dst_ty,
                                               instr->shape().element_type());
  }

  if (HloPredicateIsOp<HloOpcode::kConstant>(instr)) {
    if (!instr->shape().IsArray()) {
      return false;
    }
    // Check if instr's literal roundtrips to ty and back to its original type
    // without modification.
    PrimitiveType orig_ty = instr->shape().element_type();

    // The only reason Convert() should fail is if we don't support converting
    // from x to y, which indeed means it's not losslessly-convertible.
    absl::StatusOr<Literal> converted1 = instr->literal().Convert(dst_ty);
    if (!converted1.ok()) {
      return false;
    }
    absl::StatusOr<Literal> converted2 = converted1->Convert(orig_ty);
    if (!converted2.ok()) {
      return false;
    }

    return instr->literal() == *converted2;
  }

  if (HloPredicateIsOp<HloOpcode::kBroadcast, HloOpcode::kReshape,
                       HloOpcode::kTranspose>(instr)) {
    return IsLosslesslyConvertibleTo(instr->operand(0), dst_ty);
  }

  return false;
}

// Helpers suitable for use in m::Op().WithPredicate(...).
bool IsLosslesslyConvertibleToS8(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, S8);
}
bool IsLosslesslyConvertibleToF16(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, F16);
}

// If `conv` is a vanilla forward conv, transforms it into a
// conv-bias-activation.  If it's already a conv-bias-activation, does nothing.
//
// If `conv` is anything else, returns an error.
absl::StatusOr<HloInstruction*> EnsureIsConvBiasActivation(
    HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kCustomCall);

  if (conv->custom_call_target() == kCudnnConvBiasActivationForwardCallTarget) {
    return conv;
  }

  if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
    HloComputation* comp = conv->parent();

    const Shape& shape = conv->shape().tuple_shapes(0);
    int64_t num_output_features = shape.dimensions(
        conv->convolution_dimension_numbers().output_feature_dimension());

    // bias for integer convs is always f32, see
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    PrimitiveType bias_ty;
    if (primitive_util::IsIntegralType(shape.element_type())) {
      bias_ty = F32;
    } else {
      bias_ty = shape.element_type();
    }
    auto bias = BroadcastZeros(comp, bias_ty, {num_output_features});

    absl::InlinedVector<HloInstruction*, 3> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands.push_back(bias);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    new_conv->set_custom_call_target(kCudnnConvBiasActivationForwardCallTarget);
    comp->parent()->SetAndUniquifyInstrName(new_conv,
                                            "cudnn-conv-bias-activation");
    return new_conv;
  }

  return FailedPrecondition("Unsupported conv: %s", conv->ToString());
}

// convert<cvt_type>(gte(custom-call<conv_type>(int8_x, int8_w))) ->
// gte(custom-call<cvt_type>(int8_x, int8_w))
absl::StatusOr<bool> FuseConvertTypeIntoConv(HloComputation* comp,
                                             PrimitiveType conv_type,
                                             PrimitiveType cvt_type) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    auto tuple_elem =
        m::GetTupleElement(m::Op(&conv).WithPredicate(IsConvCustomCall), 0)
            .WithElementType(conv_type);
    auto pattern =
        m::Convert(tuple_elem.WithOneUser()).WithElementType(cvt_type);
    if (!Match(instr, pattern)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertTypeIntoConv: ", conv->ToString());
        })) {
      continue;
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(cvt_type);
    HloInstruction* new_conv =
        comp->AddInstruction(conv->CloneWithNewShape(new_shape));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_gte));

    changed = true;
  }

  return changed;
}

struct ConvConvertTypes {
  PrimitiveType convolution_type;
  PrimitiveType conversion_type;
};

// Remove convert around convolution by making the convolution-type
// (custom call) to be the same as the conversion result.
// For example: convert<float>(gte(custom-call<int32>(int8_x, int8_w))) ->
// gte(custom-call<float>(int8_x, int8_w))
absl::StatusOr<bool> FuseRemoveConvertInConv(HloComputation* comp) {
  bool changed = false;
  // Note: We are eliminating F16->F32 because it fails on internal tests.
  std::array<ConvConvertTypes, 3> types{{
      {S32, F32},
      {S8, F32},
      {F32, S8},
  }};
  for (auto [conv_type, cvt_type] : types) {
    TF_ASSIGN_OR_RETURN(bool curr_change,
                        FuseConvertTypeIntoConv(comp, conv_type, cvt_type));
    changed |= curr_change;
  }
  return changed;
}

// alpha * gte(custom-call(...)) ->
// gte(custom-call(..., backend_config={alpha})).
absl::StatusOr<bool> FuseConvAlpha(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* alpha = nullptr;

    auto pattern = m::MultiplyAnyOrder(
        m::GetTupleElement(
            &gte, m::Op(&conv).WithPredicate(IsNonDepthwiseConvCustomCall), 0)
            .WithOneUse(),
        m::Broadcast(m::ConstantEffectiveScalar(&alpha)));
    if (!Match(instr, pattern)) {
      continue;
    }

    // alpha is f32 except for f64 convs, where it's f64.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    PrimitiveType alpha_ty = gte->shape().element_type() == F64 ? F64 : F32;
    if (!IsLosslesslyConvertibleTo(alpha, alpha_ty)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();

    if (config.conv_result_scale() != 1) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvAlpha: ", conv->ToString());
        })) {
      continue;
    }

    // StreamExecutor doesn't support the alpha parameter on non-bias-activation
    // convs, so we have to upgrade `conv`.
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));

    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_conv_result_scale(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(instr, gte));

    changed = true;
  }
  return changed;
}

// The format of the serialized graph describing a sequence of ops fused
// into the cuDNN convolution Custom Call is
// "UID:[output_type]conv();UID:[output_type]op_name(operand
// UIDs);UID:[output_type]op_name(operand UIDs);..." with the convolution
// assumed to be the first op in the graph. Operand UIDs identifying ops outside
// the serialized graph are elided. Currently, multiplication and division by a
// broadcast scalar, addition, the application of various activations and the
// calculation of the maximum of the absolute value are supported.
class GraphString {
 public:
  GraphString() = default;

  bool AppendOp(std::string op_name, HloInstruction* op,
                std::vector<HloInstruction*> operands = {}) {
    return AppendOp(op_name, op, op->shape().element_type(), operands);
  }

  bool AppendOp(std::string op_name, HloInstruction* op,
                PrimitiveType element_type,
                std::vector<HloInstruction*> operands = {}) {
    // Do not add the same op multiple times.
    if (OpInGraph(op)) {
      return false;
    }

    // Insert op in front of its first use as an operand in graph_ or at the end
    // of graph_ if not an operand of another op.
    auto pos = std::find_if(
        graph_.begin(), graph_.end(), [op](OpDescriptor graph_op) -> bool {
          return std::find(graph_op.operands.begin(), graph_op.operands.end(),
                           op) != graph_op.operands.end();
        });
    pos = graph_.insert(pos, OpDescriptor{op, element_type, op_name, operands});

    // If necessary, move the operands of the op already in the graph in front
    // of the op and recursively repeat for the operands' operands. Iterator pos
    // marks the position of op in the vector and serves as an approximately
    // optimal start point when finding operands that may need to be moved. With
    // the exception of cases with multiple in-graph operands, pos is expected
    // to be graph_.end(), which causes reorder_operands to return immediately.
    std::function<void(std::vector<HloInstruction*>,
                       std::vector<OpDescriptor>::iterator)>
        reorder_operands = [&](auto operands, auto pos) {
          for (HloInstruction* operand : operands) {
            auto operand_pos = std::find_if(
                pos, graph_.end(), [operand](OpDescriptor graph_op) -> bool {
                  return operand == graph_op.instr;
                });
            if (operand_pos != graph_.end()) {
              OpDescriptor operand_desc = *operand_pos;
              graph_.erase(operand_pos);
              pos = graph_.insert(pos, operand_desc);
              reorder_operands(Operands(operand), pos);
            }
          }
        };
    reorder_operands(operands, pos);

    return true;
  }

  void ChangeDataType(PrimitiveType type) {
    DCHECK(!graph_.empty());
    graph_.back().output_type = type;
  }

  std::string Graph() const {
    std::string graph;
    for (OpDescriptor op : graph_) {
      std::vector<int64_t> operand_uids_in_graph;
      for (HloInstruction* operand : op.operands) {
        if (OpInGraph(operand)) {
          operand_uids_in_graph.push_back(operand->unique_id());
        }
      }
      graph.append(std::to_string(op.instr->unique_id()));
      graph.append(":[" +
                   primitive_util::LowercasePrimitiveTypeName(op.output_type) +
                   "]");
      graph.append(op.name);
      graph.append("(");
      graph.append(absl::StrJoin(operand_uids_in_graph, ","));
      graph.append(");");
    }
    return graph;
  }

  bool OpInGraph(HloInstruction* op, std::string op_name = "") const {
    auto op_filter = [&](OpDescriptor graph_op) -> bool {
      if (op_name.empty()) {
        return graph_op.instr->unique_id() == op->unique_id();
      } else {
        return graph_op.instr->unique_id() == op->unique_id() &&
               graph_op.name == op_name;
      }
    };
    return std::find_if(graph_.begin(), graph_.end(), op_filter) !=
           graph_.end();
  }

  std::vector<HloInstruction*> Operands(HloInstruction* op) const {
    auto op_it = std::find_if(
        graph_.begin(), graph_.end(),
        [op](OpDescriptor graph_op) -> bool { return op == graph_op.instr; });
    if (op_it != graph_.end()) {
      return op_it->operands;
    }
    return {};
  }

 private:
  struct OpDescriptor {
    HloInstruction* instr;
    PrimitiveType output_type;
    std::string name;
    std::vector<HloInstruction*> operands;
  };

  std::vector<OpDescriptor> graph_;
};

bool IsF8Type(const HloInstruction* instr) {
  return primitive_util::IsF8Type(instr->shape().element_type());
}

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsScalar(instr->shape());
}

std::optional<PrimitiveType> IsSaturatingCastToF8(HloInstruction* instr) {
  HloInstruction *op, *clamp_lower, *clamp_upper;
  if (Match(instr,
            m::Convert(
                &op,
                m::Clamp(m::Broadcast(m::ConstantScalar(&clamp_lower)), m::Op(),
                         m::Broadcast(m::ConstantScalar(&clamp_upper))))) &&
      ((op->shape().element_type() == F8E4M3FN &&
        clamp_lower->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e4m3fn>::lowest())) &&
        clamp_upper->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e4m3fn>::max()))) ||
       (op->shape().element_type() == F8E5M2 &&
        clamp_lower->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e5m2>::lowest())) &&
        clamp_upper->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e5m2>::max()))))) {
    return op->shape().element_type();
  }
  return std::nullopt;
}

// Returns whether the HLO Computation applied by `op` calculates the largest
// element.
bool AppliesMaxReduce(HloInstruction* op) {
  HloComputation* reduce_comp = op->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return ShapeUtil::IsScalar(op->shape()) &&
         ShapeUtil::IsScalar(op->operand(1)->shape()) &&
         op->operand(1)->IsConstant() &&
         op->operand(1)->literal().GetAsDouble({}) <= 0. &&
         HloPredicateIsOp<HloOpcode::kMaximum>(reduce_comp_root) &&
         HloPredicateIsOp<HloOpcode::kParameter>(
             reduce_comp_root->operand(0)) &&
         HloPredicateIsOp<HloOpcode::kParameter>(reduce_comp_root->operand(1));
}

// Recursively captures and serializes the graph of pointwise operations
// operating on the convolution.
void CaptureConvGraphRecursive(HloInstruction* instr,
                               std::vector<HloInstruction*>& operands,
                               std::vector<HloInstruction*>& aux_outputs,
                               GraphString& graph_string,
                               absl::flat_hash_set<int>& visited_instrs,
                               HloInstruction*& final_instr,
                               int& num_endpoints) {
  // Avoid visiting the same instruction more than once.
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return;
  }

  // Copy the current state in case fusion will be unsuccessful or unfavorable.
  GraphString init_graph_string = graph_string;
  std::vector<HloInstruction*> init_operands = operands;
  std::vector<HloInstruction*> init_aux_outputs = aux_outputs;
  int init_num_endpoints = num_endpoints;

  // The loop adds each user of instr that supports fusion into the cuDNN
  // convolution Custom Call to graph_string. The identification of an op is
  // followed by a recursive call of CaptureConvGraphRecursive to match and
  // potentially fuse its users. Aside from scalar reduction results, the graph
  // must have a single tensor endpoint.
  //
  // Examples:
  //
  //        C
  //       / \
  //  A - B   E - F
  //       \ /
  //        D
  //
  // The graph is fully fused since it has one endpoint.
  //
  //        C - D
  //       /
  //  A - B
  //       \
  //        E - F
  //
  // Fusion stops at B since the graph would have two endpoints.
  int num_new_users = 0;
  int num_existing_users = 0;
  for (HloInstruction* user : instr->users()) {
    HloInstruction *op0, *op1, *op2, *operand0, *operand1;
    // Add
    if (Match(user,
              m::AddAnyOrder(&op0, m::Op().Is(instr), m::Op(&operand0)))) {
      if (graph_string.AppendOp("add", op0, {instr, operand0})) {
        operands.push_back(operand0);
        ++num_new_users;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      } else {
        // Since operands only holds ops that are not part of the graph, remove
        // instr.
        operands.erase(std::remove(operands.begin(), operands.end(), instr),
                       operands.end());
        ++num_existing_users;
      }
      continue;
    }
    // Scale
    if (Match(user, m::MultiplyAnyOrder(&op0, m::Op().Is(instr),
                                        m::Broadcast(m::Op(&operand0)))) &&
        ShapeUtil::IsScalar(operand0->shape())) {
      if (graph_string.AppendOp("scale", op0, {instr, operand0})) {
        operands.push_back(operand0);
        ++num_new_users;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      }
      continue;
    }
    // Inverse Scale
    if (Match(user, m::Divide(&op0, m::Op().Is(instr),
                              m::Broadcast(m::Op(&operand0)))) &&
        ShapeUtil::IsScalar(operand0->shape())) {
      if (graph_string.AppendOp("invscale", op0, {instr, operand0})) {
        operands.push_back(operand0);
        ++num_new_users;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      }
      continue;
    }
    // ReLU
    if (Match(user, m::MaximumAnyOrder(&op0, m::Op().Is(instr),
                                       m::Broadcast(m::ConstantScalar(0))))) {
      if (graph_string.AppendOp("relu", op0, {instr})) {
        ++num_new_users;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      }
      continue;
    }
    //  Maximum of the absolute value (Amax) following ReLU (elided Abs)
    if (Match(user, m::Reduce(&op0, m::Op().Is(instr), m::Op())) &&
        graph_string.OpInGraph(instr, "relu") && AppliesMaxReduce(op0)) {
      if (graph_string.AppendOp("amax", op0, {instr})) {
        aux_outputs.push_back(op0);
        ++num_new_users;
      }
      continue;
    }
    // ReLU6, equivalently represented in the cuDNN graph as min(ReLU, 6)
    if (Match(
            user,
            m::Clamp(
                &op1,
                m::Broadcast(&op0, m::ConstantEffectiveScalar(0)).WithOneUser(),
                m::Op().Is(instr),
                m::Broadcast(&operand0, m::ConstantEffectiveScalar(6))
                    .WithOneUser()))) {
      if (!graph_string.OpInGraph(op0) && !graph_string.OpInGraph(op1)) {
        graph_string.AppendOp("relu", op0, {instr});
        graph_string.AppendOp("min", op1, {op0, operand0});
        ++num_new_users;
        operands.push_back(operand0);
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      }
      continue;
    }
    // ELU
    if (Match(user,
              m::Select(&op0,
                        m::Compare(m::Op().Is(instr),
                                   m::Broadcast(m::ConstantEffectiveScalar(0)))
                            .WithComparisonDirection(ComparisonDirection::kGt),
                        m::Op().Is(instr), m::Expm1(m::Op().Is(instr))))) {
      if (graph_string.AppendOp("elu", op0, {instr})) {
        num_new_users += 3;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr, num_endpoints);
      }
      continue;
    }

    // The following patterns match the user of `user`.
    if (user->user_count() == 1) {
      HloInstruction* users_user = user->users()[0];
      // Convert with Clamp to FP8 types
      std::optional<PrimitiveType> f8_type = IsSaturatingCastToF8(users_user);
      if (f8_type.has_value()) {
        graph_string.ChangeDataType(f8_type.value());
        ++num_new_users;
        CaptureConvGraphRecursive(users_user, operands, aux_outputs,
                                  graph_string, visited_instrs, final_instr,
                                  num_endpoints);
        continue;
      }
      // Maximum of the absolute value (Amax)
      if (Match(users_user,
                m::Reduce(&op0, m::Abs(m::Op().Is(instr)), m::Op())) &&
          AppliesMaxReduce(op0)) {
        if (graph_string.AppendOp("amax", op0, {instr})) {
          aux_outputs.push_back(op0);
          ++num_new_users;
        }
        continue;
      }
      // Leaky ReLU, equivalently represented in the cuDNN graph as the maximum
      // of the value and the slope times the minimum of zero and the value,
      // max(x, alpha * min(0, x))
      if (Match(
              users_user,
              m::Select(&op2,
                        m::Compare(&op0, m::Op().Is(instr),
                                   m::Broadcast(&operand0,
                                                m::ConstantEffectiveScalar(0)))
                            .WithComparisonDirection(ComparisonDirection::kGt)
                            .WithOneUser(),
                        m::Op().Is(instr),
                        m::MultiplyAnyOrder(
                            &op1, m::Op().Is(instr),
                            m::Broadcast(m::ConstantEffectiveScalar(&operand1)))
                            .WithOneUser()))) {
        if (!graph_string.OpInGraph(op0) && !graph_string.OpInGraph(op1) &&
            !graph_string.OpInGraph(op2)) {
          graph_string.AppendOp("min", op0, op2->shape().element_type(),
                                {instr, operand0});
          graph_string.AppendOp("scale", op1, {op0, operand1});
          graph_string.AppendOp("max", op2, {op1, instr});
          num_new_users += 3;
          operands.push_back(operand0);
          operands.push_back(operand1);
          CaptureConvGraphRecursive(users_user, operands, aux_outputs,
                                    graph_string, visited_instrs, final_instr,
                                    num_endpoints);
        }
        continue;
      }
    }
  }

  // instr is an endpoint when the number of users eligible for fusion is less
  // than the total number of users or when instr has no users.
  if (num_new_users + num_existing_users < instr->user_count() ||
      instr->user_count() == 0) {
    final_instr = instr;
    ++num_endpoints;
  }

  // Since the cuDNN graph cannot have more than one endpoint, do not fuse
  // into the cuDNN convolution Custom Call and roll back the graph when there
  // are multiple endpoints. If the resulting graph still has more than one
  // endpoint, the recursive caller will continue to roll back the graph.
  if (num_endpoints > 1) {
    graph_string = std::move(init_graph_string);
    operands = init_operands;
    aux_outputs = init_aux_outputs;
    final_instr = instr;
    // Reverting the graph removes any users of instr from the cuDNN graph and
    // makes instr an endpoint.
    num_endpoints = init_num_endpoints + 1;
  }
}

// Captures in a GraphString the subgraph of pointwise operations operating on
// the convolution that will be fused into the cuDNN convolution Custom Call.
absl::StatusOr<
    std::tuple<std::vector<HloInstruction*>, std::vector<HloInstruction*>,
               GraphString, HloInstruction*>>
CaptureConvGraph(HloInstruction* instr, HloInstruction* convolution,
                 HloInstruction* wide_input, HloInstruction* wide_filter,
                 HloInstruction* input_scale, HloInstruction* filter_scale,
                 bool x_mult_scale, bool w_mult_scale) {
  GraphString graph_string;
  graph_string.AppendOp("conv", instr);

  // Shift the scaling of the input and filter to the output of the convolution.
  HloInstruction *input_scaled_conv, *filter_scaled_conv;
  if (input_scale) {
    TF_RETURN_IF_ERROR(convolution->ReplaceOperandWith(0, wide_input));
    HloInstruction* bcast_input_scale = instr->AddInstruction(
        HloInstruction::CreateBroadcast(instr->shape(), input_scale, {}));
    input_scaled_conv = instr->AddInstruction(HloInstruction::CreateBinary(
        instr->shape(),
        x_mult_scale ? HloOpcode::kMultiply : HloOpcode::kDivide, instr,
        bcast_input_scale));
    TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(input_scaled_conv));
  }
  if (filter_scale) {
    TF_RETURN_IF_ERROR(convolution->ReplaceOperandWith(1, wide_filter));
    HloInstruction* bcast_filter_scale = instr->AddInstruction(
        HloInstruction::CreateBroadcast(instr->shape(), filter_scale, {}));
    filter_scaled_conv = instr->AddInstruction(HloInstruction::CreateBinary(
        instr->shape(),
        w_mult_scale ? HloOpcode::kMultiply : HloOpcode::kDivide,
        input_scale ? input_scaled_conv : instr, bcast_filter_scale));
    TF_RETURN_IF_ERROR((input_scale ? input_scaled_conv : instr)
                           ->ReplaceAllUsesWith(filter_scaled_conv));
  }

  std::vector<HloInstruction*> operands, aux_outputs;
  absl::flat_hash_set<int> visited_instrs;
  HloInstruction* final_instr;
  int num_endpoints = 0;
  CaptureConvGraphRecursive(instr, operands, aux_outputs, graph_string,
                            visited_instrs, final_instr, num_endpoints);
  return std::make_tuple(operands, aux_outputs, graph_string, final_instr);
}

// Matches convolutions operating on FP8 inputs and filters and rewrites into a
// ForwardGraph Custom Call. For scaled FP8 convolutions on Hopper systems, the
// following steps are elided and rewritten into a ForwardGraph Custom Call:
//
// 1. Cast the filter and input from FP8 to a wider type such as FP16 or FP32.
// 2. Optionally unscale the filter and input by multiplying or dividing by
// scalars.
// 3. Evaluate the convolution based on the scaled filter and input.
// 4. Apply a series of elementwise transformations.
// 5. Optionally calculate the maximum of the absolute of the result.
// 6. Optionally cast the output back to FP8.
absl::StatusOr<bool> F8GraphConv(HloComputation* comp,
                                 se::CudaComputeCapability cc,
                                 se::dnn::VersionInfo dnn_version,
                                 const se::SemanticVersion& toolkit_version) {
  bool changed = false;

  if (dnn_version < se::dnn::VersionInfo(8, 9, 0)) {
    return false;
  }
  if (toolkit_version < se::SemanticVersion{12, 0, 0}) {
    return false;
  }
  if (!cc.IsAtLeast(se::CudaComputeCapability::kHopper)) {
    return false;
  }
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *convolution, *gte, *input, *filter,
        *input_scale = nullptr, *filter_scale = nullptr,
        *input_scale_op = nullptr, *filter_scale_op = nullptr,
        *wide_input = nullptr, *wide_filter = nullptr;

    auto conv_operand_maybe_scaled = [](HloInstruction** operand,
                                        HloInstruction** wide_operand,
                                        HloInstruction** scale_op,
                                        HloInstruction** scale) {
      return m::AnyOf<HloInstruction>(
          m::Op(operand).WithPredicate(IsF8Type),
          m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
          m::Divide(
              scale_op,
              m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
              m::Broadcast(m::Op(scale).WithPredicate(IsScalar))),
          m::MultiplyAnyOrder(
              scale_op,
              m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
              m::Broadcast(m::Op(scale).WithPredicate(IsScalar))));
    };

    // TODO(philipphack): Consider allowing ops between dequantization and
    // convolution.
    auto pattern = m::GetTupleElement(
        &gte,
        m::CustomCall(
            &convolution,
            conv_operand_maybe_scaled(&input, &wide_input, &input_scale_op,
                                      &input_scale),
            conv_operand_maybe_scaled(&filter, &wide_filter, &filter_scale_op,
                                      &filter_scale))
            .WithPredicate(IsConvCustomCall),
        0);
    if (Match(instr, pattern)) {
      if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
            return absl::StrCat("F8GraphConv: ", convolution->ToString());
          })) {
        continue;
      }

      std::vector<HloInstruction*> operands, aux_outputs;
      GraphString graph_string;
      HloInstruction* final_instr;

      TF_ASSIGN_OR_RETURN(
          std::tie(operands, aux_outputs, graph_string, final_instr),
          CaptureConvGraph(
              instr, convolution, wide_input, wide_filter, input_scale,
              filter_scale,
              input_scale_op
                  ? HloPredicateIsOp<HloOpcode::kMultiply>(input_scale_op)
                  : false,
              filter_scale_op
                  ? HloPredicateIsOp<HloOpcode::kMultiply>(filter_scale_op)
                  : false));
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          convolution->backend_config<GpuBackendConfig>());
      CudnnConvBackendConfig& config =
          *gpu_config.mutable_cudnn_conv_backend_config();

      config.set_serialized_graph(graph_string.Graph());
      operands.insert(operands.begin(), input);
      operands.insert(operands.begin() + 1, filter);

      std::vector<Shape> output_shapes;
      output_shapes.emplace_back(ShapeUtil::ChangeElementType(
          ShapeUtil::GetTupleElementShape(convolution->shape(), 0),
          final_instr->shape().element_type()));
      for (HloInstruction* aux_output : aux_outputs) {
        output_shapes.emplace_back(aux_output->shape());
      }
      output_shapes.emplace_back(
          ShapeUtil::GetTupleElementShape(convolution->shape(), 1));

      HloInstruction* new_convolution =
          comp->AddInstruction(convolution->CloneWithNewOperands(
              ShapeUtil::MakeTupleShape(output_shapes), operands));

      new_convolution->set_custom_call_target(kCudnnConvForwardGraphCallTarget);
      TF_RETURN_IF_ERROR(new_convolution->set_backend_config(gpu_config));
      TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                          MakeGetTupleElementHlo(new_convolution, 0));
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(final_instr, new_gte));

      for (int i = 0; i < aux_outputs.size(); ++i) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                            MakeGetTupleElementHlo(new_convolution, i + 1));
        TF_RETURN_IF_ERROR(comp->ReplaceInstruction(aux_outputs[i], new_gte));
      }

      changed = true;
    }
  }
  return changed;
}

absl::StatusOr<bool> FuseBiasOrSideInput(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* addend = nullptr;

    auto pattern = m::AddAnyOrder(
        m::GetTupleElement(&gte,
                           m::Op(&conv)
                               .WithPredicate(IsNonDepthwiseConvCustomCall)
                               .WithOneUse(),
                           0)
            .WithOneUse(),
        m::Op(&addend));
    if (!Match(instr, pattern)) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseBiasOrSideInput: ", conv->ToString());
        })) {
      continue;
    }

    // If it's a vanilla forward conv, upgrade it to a bias-activation conv.  We
    // only want to do this if the fusion will succeed, but we're guaranteed
    // that it will, because the only reason we'll bail at this point is if
    // !can_accept_bias && !can_accept_side_input, and our shiny new
    // bias-activation conv will be able to accept both.
    if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
      TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    }

    // Can't fuse bias or side-input if the conv already has a relu (or other
    // activation), because bias and side-input are added before the activation
    // is applied.
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    // Does `conv` already have a (nonzero) bias?  Does it already have a
    // side_input?
    bool can_accept_bias =
        Match(conv->operand(2), m::Broadcast(m::ConstantEffectiveScalar(0)));
    bool can_accept_side_input = conv->operand_count() < 4;

    // The addend can be fused as a bias if
    //  - it is 1D broadcasted in the output feature dimension, and
    //  - it is losslessly-convertible to the correct type (f32 for s8/f32/u32
    //    convs, and conv_ty for floating-point convs)
    PrimitiveType conv_ty = gte->shape().element_type();
    PrimitiveType bias_ty =
        primitive_util::IsFloatingPointType(conv_ty) ? conv_ty : F32;
    bool addend_may_be_rank1_bias =
        HloPredicateIsOp<HloOpcode::kBroadcast>(addend) &&
        addend->dimensions().size() == 1 &&
        addend->dimensions(0) ==
            conv->convolution_dimension_numbers().output_feature_dimension() &&
        IsLosslesslyConvertibleTo(addend, bias_ty);

    bool addend_may_be_rank0_bias =
        HloPredicateIsOp<HloOpcode::kBroadcast>(addend) &&
        addend->dimensions().empty() &&
        IsLosslesslyConvertibleTo(addend, bias_ty);

    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    if (can_accept_bias && addend_may_be_rank1_bias) {
      new_operands[2] = MakeConvertToHlo(addend->mutable_operand(0), bias_ty,
                                         &addend->operand(0)->metadata());
    } else if (can_accept_bias && addend_may_be_rank0_bias) {
      new_operands[2] = MakeBroadcastHlo(
          MakeConvertToHlo(addend->mutable_operand(0), bias_ty,
                           &addend->operand(0)->metadata()),
          /*broadcast_dimensions=*/{},
          /*result_shape_bounds=*/
          {gte->shape().dimensions(conv->convolution_dimension_numbers()
                                       .output_feature_dimension())});
    } else if (can_accept_side_input) {
      CHECK_EQ(new_operands.size(), 3);
      new_operands.push_back(addend);
      config.set_side_input_scale(1);
    } else {
      // Can't fuse; this op already has a bias and a side-input.
      continue;
    }

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

// custom-call(..., alpha * side_input) ->
// custom-call(..., side_input, backend_config={alpha}).
//
// We also have to support the more complicated case of
//
//   custom-call(..., reshape(side_input * alpha)) -->
//   custom-call(..., reshape(side_input), backend_config={alpha}),
//
// where `reshape` can be an arbitrary chain of reshapes+transposes.  This idiom
// is created by the ReshapeMover pass.
absl::StatusOr<bool> FuseSideInputAlpha(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv;
    HloInstruction* side_input;
    auto pattern = m::Op(&conv)
                       .WithPredicate(IsConvCustomCall)
                       .WithOperand(3, m::Op(&side_input));
    if (!Match(instr, pattern)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.side_input_scale() != 1) {
      continue;
    }

    // Given side_input, pattern match the following (working from bottom up).
    //
    // before_reshape = multiply(base, broadcast(alpha))
    // side_input = chain_of_reshapes_and_transposes(before_reshape)
    //
    // where alpha is a scalar constant.
    //
    // alpha is f32 except for f64 convs, where it's f64.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    HloInstruction* before_reshape = side_input;
    while (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kTranspose>(
        before_reshape)) {
      before_reshape = before_reshape->mutable_operand(0);
    }

    PrimitiveType conv_ty = conv->shape().tuple_shapes(0).element_type();
    PrimitiveType alpha_ty = conv_ty == F64 ? F64 : F32;
    HloInstruction* base;
    HloInstruction* alpha;
    if (!Match(
            before_reshape,
            m::MultiplyAnyOrder(
                m::Op(&base),
                m::Broadcast(m::ConstantEffectiveScalar(&alpha).WithPredicate(
                    [&](const HloInstruction* instr) {
                      return IsLosslesslyConvertibleTo(instr, alpha_ty);
                    }))))) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseSideInputAlpha: ", conv->ToString());
        })) {
      continue;
    }

    // Rewrite conv's operand 3 to
    //
    //   chain_of_reshapes_and_transposes(before_reshape).
    //
    // and store alpha in the conv's backend config.
    //
    // We're going to do something bad here: We aren't going to check that the
    // chain of reshapes/transposes has one use, so we're potentially
    // duplicating all these instructions (once with alpha and once without).
    //
    // This is justified because
    //
    //  - duplicating reshapes/transposes shouldn't be "that bad" -- these
    //    instructions can usually be fused, and
    //
    //  - *not* fusing alpha can be catastrophic.  For s8->s8 convolutions, the
    //    side-input must be s8.  But the product side_input * alpha is f32, so
    //    we can only see that side-input is s8 if we fuse alpha. IOW not fusing
    //    alpha means we'll run this s8->s8 conv as s8->f32, which is *much*
    //    slower than some extra transposes.

    // Recursively clone chain_of_reshapes_and_transposes until we get to
    // `before_reshape`, at which point we skip the multiply(base, alpha) and
    // just return base.
    std::function<HloInstruction*(const HloInstruction*)> clone =
        [&](const HloInstruction* instr) {
          if (instr == before_reshape) {
            return base;
          }
          CHECK((HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kTranspose>(
              instr)))
              << "Must be reshape or transpose: " << instr->ToString();
          return comp->AddInstruction(instr->CloneWithNewOperands(
              instr->shape(), {clone(instr->operand(0))}));
        };
    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[3] = clone(side_input);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());

    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_side_input_scale(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));

    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseElu(HloComputation* comp,
                             se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }

  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte1, *gte2, *gte3;
    HloInstruction* conv;
    HloInstruction* expm1;

    if (!Match(instr,
               m::Select(m::Compare(m::GetTupleElement(&gte1, m::Op()),
                                    m::Broadcast(m::ConstantEffectiveScalar(0)))
                             .WithComparisonDirection(ComparisonDirection::kGt)
                             .WithOneUse(),
                         m::GetTupleElement(
                             &gte2,
                             m::Op(&conv)
                                 .WithPredicate(IsNonDepthwiseConvCustomCall)
                                 .WithOneUse(),
                             /*tuple_index=*/0)
                             // TODO(jlebar): Why only fp16?
                             .WithElementType(F16),
                         m::Op(&expm1)
                             .WithOpcode(HloOpcode::kExpm1)
                             .WithOperand(0, m::GetTupleElement(&gte3, m::Op()))
                             .WithOneUse()))) {
      continue;
    }

    // The three GTEs should be the same, and these should be the only uses.
    if (gte1 != gte2 || gte2 != gte3 || gte1->user_count() != 3) {
      continue;
    }

    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseElu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kElu);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte1));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseRelu(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte;
    HloInstruction* conv;
    if (!Match(instr,
               m::MaximumAnyOrder(
                   m::Broadcast(m::ConstantEffectiveScalar(0)),
                   m::GetTupleElement(
                       &gte, m::Op(&conv)
                                 .WithPredicate(IsNonDepthwiseConvCustomCall)
                                 .WithOneUse())
                       .WithOneUse()))) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseRelu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kRelu);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseRelu6(HloComputation* comp,
                               se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }

  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte, *conv;
    if (!Match(
            instr,
            m::Clamp(m::Broadcast(m::ConstantEffectiveScalar(0)),
                     m::GetTupleElement(
                         &gte, m::Op(&conv)
                                   .WithPredicate(IsNonDepthwiseConvCustomCall)
                                   .WithOneUse())
                         // TODO(jlebar): Why only fp16?
                         .WithElementType(F16)
                         .WithOneUse(),
                     m::Broadcast(m::ConstantEffectiveScalar(6))))) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseRelu6: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kRelu6);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseLeakyRelu(HloComputation* comp,
                                   se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }

  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte1, *gte2, *gte3, *conv, *alpha;
    if (!Match(instr,
               m::Select(
                   m::Compare(m::GetTupleElement(&gte1, m::Op()),
                              m::Broadcast(m::ConstantEffectiveScalar(0)))
                       .WithComparisonDirection(ComparisonDirection::kGt)
                       .WithOneUse(),
                   m::GetTupleElement(
                       &gte2, m::Op(&conv)
                                  .WithPredicate(IsNonDepthwiseConvCustomCall)
                                  .WithOneUse())
                       // TODO(jlebar): Why only fp16?
                       .WithElementType(F16),
                   m::Multiply(m::GetTupleElement(&gte3, m::Op()),
                               m::Broadcast(m::ConstantEffectiveScalar(&alpha)))
                       .WithOneUse()))) {
      continue;
    }

    // The three GTEs should be the same, and these should be the only uses.
    if (gte1 != gte2 || gte2 != gte3 || gte1->user_count() != 3) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseLeakyRelu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kLeakyRelu);
    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_leakyrelu_alpha(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte1));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseConvertToF16(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;

    auto f32_convertible_to_f16_pat =
        m::Op().WithElementType(F32).WithPredicate(
            IsLosslesslyConvertibleToF16);
    if (!MatchAndLogIfFailed(
            instr, "f16 conv",
            m::Convert(
                m::GetTupleElement(
                    &gte,
                    m::Op(&conv)
                        .WithPredicate(IsConvCustomCall)
                        .WithOperand(0, f32_convertible_to_f16_pat)
                        .WithOperand(1, f32_convertible_to_f16_pat)
                        .WithOperandIfPresent(2, f32_convertible_to_f16_pat)
                        .WithOperandIfPresent(3, f32_convertible_to_f16_pat),
                    0)
                    .WithOneUse())
                .WithElementType(F16),
            VLOG_IS_ON(3),
            m::Op().WithOperand(0, m::GetTupleElement(m::Op().WithPredicate(
                                       IsConvCustomCall))))) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToF16: ", conv->ToString());
        })) {
      continue;
    }

    VLOG(2) << "Matched fp16 conv: " << conv->ToString();

    // In fp16 convs, all operands, including `bias`, must be fp16.  This is
    // different from int8 convs, where the bias is fp32.  See table of
    // supported datatypes at
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    absl::InlinedVector<HloInstruction*, 4> new_operands;
    for (HloInstruction* operand : conv->operands()) {
      new_operands.push_back(
          MakeConvertToHlo(operand, F16, &operand->metadata()));
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(F16);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FuseConvertToS8(HloComputation* comp,
                                     se::GpuComputeCapability cc) {
  if (IsROCm(cc)) return false;
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;

    auto conv_pattern =
        m::Op(&conv)
            .WithPredicate(IsConvCustomCall)
            .WithOperand(0, m::Op().WithPredicate(IsLosslesslyConvertibleToS8))
            .WithOperand(1, m::Op().WithPredicate(IsLosslesslyConvertibleToS8));

    PrimitiveType conv_output_ty;
    if (MatchAndLogIfFailed(
            instr, "s8->s8 conv with clamp",
            m::Convert(m::Clamp(m::Broadcast(m::ConstantEffectiveScalar(-128)),
                                m::GetTupleElement(
                                    &gte,
                                    conv_pattern.WithOperandIfPresent(
                                        3, m::Op().WithPredicate(
                                               IsLosslesslyConvertibleToS8)),
                                    0)
                                    .WithOneUse(),
                                m::Broadcast(m::ConstantEffectiveScalar(127))))
                .WithElementType(S8),
            VLOG_IS_ON(3),
            m::Convert(m::Clamp(m::Op(),
                                m::GetTupleElement(
                                    m::Op().WithPredicate(IsConvCustomCall)),
                                m::Op()))
                .WithElementType(S8))) {
      conv_output_ty = S8;
    } else if (MatchAndLogIfFailed(
                   instr, "s8->s8 conv without clamp",
                   m::Convert(m::GetTupleElement(
                                  &gte,
                                  conv_pattern.WithOperandIfPresent(
                                      3, m::Op().WithPredicate(
                                             IsLosslesslyConvertibleToS8)),
                                  0)
                                  .WithElementType(F32)
                                  .WithOneUse())
                       .WithElementType(S8),
                   VLOG_IS_ON(3),
                   m::Convert(m::GetTupleElement(
                                  m::Op().WithPredicate(IsConvCustomCall)))
                       .WithElementType(S8))) {
      conv_output_ty = S8;
    } else if (MatchAndLogIfFailed(
                   instr, "s8->f32 conv",
                   m::GetTupleElement(&gte,
                                      conv_pattern.WithOperandIfPresent(
                                          3, m::Op().WithElementType(F32)),
                                      0)
                       .WithElementType(F32),
                   VLOG_IS_ON(3),
                   m::GetTupleElement(m::Op().WithPredicate(IsConvCustomCall))
                       .WithElementType(F32))) {
      conv_output_ty = F32;
    } else {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToS8: ", conv->ToString());
        })) {
      continue;
    }

    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[0] =
        MakeConvertToHlo(new_operands[0], S8, &new_operands[0]->metadata());
    new_operands[1] =
        MakeConvertToHlo(new_operands[1], S8, &new_operands[1]->metadata());
    // Don't convert bias (operand 2); it's always f32 for s8 ops in cudnn.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    if (new_operands.size() >= 4) {
      // side-input always matches conv output type.  We checked in the patterns
      // above that it's losslessly-convertible to this type.
      new_operands[3] = MakeConvertToHlo(new_operands[3], conv_output_ty,
                                         &new_operands[3]->metadata());
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(conv_output_ty);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

absl::Status CheckNoIllegalIntegerConvs(HloComputation* comp) {
  auto is_integral_not_s8 = [](const Shape& s) {
    return primitive_util::IsIntegralType(s.element_type()) &&
           s.element_type() != S8;
  };

  std::vector<HloInstruction*> bad_convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (!IsConvCustomCall(instr)) {
      continue;
    }
    if (is_integral_not_s8(instr->shape().tuple_shapes(0)) ||
        is_integral_not_s8(instr->operand(0)->shape()) ||
        is_integral_not_s8(instr->operand(1)->shape()) ||
        (instr->operand_count() >= 4 &&
         is_integral_not_s8(instr->operand(3)->shape()))) {
      bad_convs.push_back(instr);
    }
  }

  if (bad_convs.empty()) {
    return absl::OkStatus();
  }

  return Unimplemented(
      R"(
Can't lower one or more integer convolutions to idioms supported by CuDNN.

CuDNN integer convolutions must have:

  - s8 input and filter,
  - f32 bias (if present),
  - s8 or f32 output, and
  - s8 side_input (if present) if output is s8.

For each of the unsupported convs below, we weren't able to lower one of the
operands or the output to the appropriate type.

See specific HLO idioms in cudnn_fused_conv_rewriter.h, and see cudnn semantics:

https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward and
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters

Unsupported convs:
%s

******* Full HLO module *******
%s
)",
      absl::StrJoin(bad_convs, "\n",
                    [](std::string* out, HloInstruction* instr) {
                      absl::StrAppend(out, " - ", instr->ToString());
                    }),
      comp->parent()->ToString());
}

void VlogStats(HloModule* module) {
  if (!VLOG_IS_ON(1)) {
    return;
  }

  VLOG(1) << "Results of CudnnFusedConvRewriter for " << module->name();
  absl::flat_hash_map<std::string, int> stats;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!Match(instr, m::Op().WithPredicate(IsConvCustomCall))) {
        continue;
      }

      VLOG(3) << instr->ToString();

      if (instr->custom_call_target() == kCudnnConvForwardCallTarget) {
        ++stats["01 non-fused forward convs"];
      } else if (instr->custom_call_target() ==
                 kCudnnConvBiasActivationForwardCallTarget) {
        ++stats["02 fused forward convs"];
      }

      PrimitiveType conv_in_ty = instr->operand(0)->shape().element_type();
      PrimitiveType conv_out_ty = instr->shape().tuple_shapes(0).element_type();
      if (conv_in_ty == F32) {
        ++stats["10 f32 convs"];
      } else if (conv_in_ty == F16) {
        ++stats["11 f16 convs"];
      } else if (conv_in_ty == S8) {
        if (conv_out_ty == S8) {
          ++stats["12 s8->s8 convs"];
        } else if (conv_out_ty == F32) {
          ++stats["13 s8->f32 convs"];
        } else {
          LOG(ERROR) << "Unexpected conv: " << instr->ToString();
        }
      }

      if (instr->operand_count() > 2) {
        ++stats["20 convs with bias"];
        if (Match(instr->operand(2),
                  m::Broadcast(m::ConstantEffectiveScalar(0)))) {
          ++stats["21 convs with 0 bias"];
        }
      }
      if (instr->operand_count() > 3) {
        ++stats["22 convs with side-input"];
      }

      auto gpu_config = instr->backend_config<GpuBackendConfig>();
      if (!gpu_config.ok()) {
        LOG(ERROR) << "Couldn't parse backend config for " << instr->ToString();
        continue;
      }
      const CudnnConvBackendConfig& config =
          gpu_config->cudnn_conv_backend_config();
      if (config.conv_result_scale() != 1) {
        ++stats["30 convs with result scale"];
      }
      if (config.side_input_scale() != 0 && config.side_input_scale() != 1) {
        ++stats["31 convs with side-input scale"];
      }
      ++stats[absl::StrCat(
          "32 convs with activation mode ",
          se::dnn::ActivationMode_Name(config.activation_mode()))];
    }
  }

  std::vector<std::pair<std::string, int>> stats_sorted(stats.begin(),
                                                        stats.end());
  absl::c_sort(stats_sorted);
  for (const auto& kv : stats_sorted) {
    VLOG(1) << absl::StreamFormat("%4d %s", kv.second,
                                  absl::string_view(kv.first).substr(3));
  }
}

}  // namespace

absl::StatusOr<bool> CudnnFusedConvRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    bool changed = false;
    // Rewrite FP8 convolutions and supported adjacent pointwise ops into a
    // ForwardGraph Custom Call.
    if (!IsROCm(compute_capability_)) {
      auto cc = std::get<se::CudaComputeCapability>(compute_capability_);
      TF_ASSIGN_OR_RETURN(
          changed, F8GraphConv(comp, cc, dnn_version_, toolkit_version_));
      if (changed) {
        return changed;
      }
    }
    // Fuse "inside out" starting with the operations closest to the conv.
    TF_ASSIGN_OR_RETURN(changed, FuseRemoveConvertInConv(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvAlpha(comp));
    any_changed |= changed;

    // s8 convs' bias and side-input appear before conversion to s8.
    //
    // Run FuseBiasOrSideInput twice, so we get both the bias and the side
    // input, if both are present.
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;

    // Relu might appear before or after convert-to-f16/s8, so we check in both
    // cases.
    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseElu(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu6(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseLeakyRelu(comp, compute_capability_));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvertToF16(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvertToS8(comp, compute_capability_));
    any_changed |= changed;

    // f16 convs' bias+side-input can appear before or after conversion to f16.
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseElu(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu6(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseLeakyRelu(comp, compute_capability_));
    any_changed |= changed;

    // Check that we don't have any convs outputting integer types other than
    // s8 - cudnn does not support these.  They should have been transformed to
    // int8->int8 or int8->float above.
    TF_RETURN_IF_ERROR(CheckNoIllegalIntegerConvs(comp));
  }

  VlogStats(module);

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
