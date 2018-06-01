/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <algorithm>
#include <deque>
#include <ostream>
#include <set>
#include <unordered_set>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using tensorflow::str_util::CEscape;
using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

/* static */
StatusOr<std::unique_ptr<HloInstruction>> HloInstruction::CreateFromProto(
    const HloInstructionProto& proto,
    const tensorflow::gtl::FlatMap<int64, HloInstruction*>& instruction_map,
    const tensorflow::gtl::FlatMap<int64, HloComputation*>& computation_map) {
  TF_RET_CHECK(!proto.opcode().empty());
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(proto.opcode()));
  TF_RET_CHECK(proto.has_shape());

  auto instruction = WrapUnique(new HloInstruction(opcode, proto.shape()));
  for (const int64 operand_id : proto.operand_ids()) {
    TF_RET_CHECK(ContainsKey(instruction_map, operand_id))
        << "No instruction with id " << operand_id;
    instruction->AppendOperand(instruction_map.at(operand_id));
  }
  for (const int64 predecessor_id : proto.control_predecessor_ids()) {
    TF_RET_CHECK(ContainsKey(instruction_map, predecessor_id))
        << "No instruction with id " << predecessor_id;
    TF_RETURN_IF_ERROR(instruction_map.at(predecessor_id)
                           ->AddControlDependencyTo(instruction.get()));
  }

  // In the proto, fused computations are held exclusively within the
  // HloInstructionProto and do not appear as an HloComputationProto within the
  // HloModuleProto.
  if (instruction->opcode() == HloOpcode::kFusion) {
    TF_RET_CHECK(!proto.fusion_kind().empty());
    TF_ASSIGN_OR_RETURN(instruction->fusion_kind_,
                        StringToFusionKind(proto.fusion_kind()));

    // Find the fused computation and set its fusion instruction.
    TF_RET_CHECK(proto.called_computation_ids_size() == 1)
        << "Expect 1 called computation for fusion instruction, but sees "
        << proto.called_computation_ids_size();
    const int64 fusion_id = proto.called_computation_ids(0);
    auto* fused_computation = FindPtrOrNull(computation_map, fusion_id);
    TF_RET_CHECK(fused_computation != nullptr)
        << "No fusion computation with id " << fusion_id;
    fused_computation->SetFusionInstruction(instruction.get());
    instruction->called_computations_.push_back(fused_computation);
  } else {
    for (const int64 computation_id : proto.called_computation_ids()) {
      TF_RET_CHECK(ContainsKey(computation_map, computation_id))
          << "No computation with id " << computation_id;
      instruction->called_computations_.push_back(
          computation_map.at(computation_id));
    }
  }

  if (instruction->opcode() == HloOpcode::kTrace) {
    TF_RET_CHECK(instruction->operands().size() == 1)
        << "Trace instruction should have 1 operand but sees "
        << instruction->operands().size();
    instruction->mutable_operand(0)->set_tracing(instruction.get());
  }

  TF_RET_CHECK(!proto.name().empty());
  instruction->name_ = proto.name();

  instruction->metadata_ = proto.metadata();
  instruction->backend_config_ = proto.backend_config();
  if (proto.has_literal()) {
    TF_ASSIGN_OR_RETURN(instruction->literal_,
                        Literal::CreateFromProto(proto.literal()));
  }
  instruction->parameter_number_ = proto.parameter_number();

  instruction->tuple_index_ = proto.tuple_index();
  for (int64 dimension : proto.dimensions()) {
    instruction->dimensions_.push_back(dimension);
  }
  if (proto.has_window()) {
    instruction->window_ = MakeUnique<Window>(proto.window());
  }
  if (proto.has_convolution_dimension_numbers()) {
    instruction->convolution_dimension_numbers_ =
        MakeUnique<ConvolutionDimensionNumbers>(
            proto.convolution_dimension_numbers());
  }
  if (proto.has_dot_dimension_numbers()) {
    instruction->dot_dimension_numbers_ =
        MakeUnique<DotDimensionNumbers>(proto.dot_dimension_numbers());
  }
  for (const HloInstructionProto::SliceDimensions& slice_dimensions :
       proto.slice_dimensions()) {
    instruction->slice_starts_.push_back(slice_dimensions.start());
    instruction->slice_limits_.push_back(slice_dimensions.limit());
    instruction->slice_strides_.push_back(slice_dimensions.stride());
  }
  instruction->exponent_bits_ = proto.exponent_bits();
  instruction->mantissa_bits_ = proto.mantissa_bits();
  for (int64 dynamic_slice_size : proto.dynamic_slice_sizes()) {
    instruction->dynamic_slice_sizes_.push_back(dynamic_slice_size);
  }
  if (proto.has_padding_config()) {
    instruction->padding_config_ =
        MakeUnique<PaddingConfig>(proto.padding_config());
  }
  instruction->outfeed_config_ = proto.outfeed_config();
  instruction->distribution_ = proto.distribution();
  instruction->epsilon_ = proto.epsilon();
  instruction->feature_index_ = proto.feature_index();
  instruction->channel_id_ = proto.channel_id();
  instruction->infeed_config_ = proto.infeed_config();
  instruction->custom_call_target_ = proto.custom_call_target();
  instruction->outfeed_shape_ = proto.outfeed_shape();
  instruction->fft_type_ = proto.fft_type();
  for (int64 fft_len : proto.fft_length()) {
    instruction->fft_length_.push_back(fft_len);
  }

  if (proto.has_sharding()) {
    TF_ASSIGN_OR_RETURN(const auto& sharding,
                        HloSharding::FromProto(proto.sharding()));
    instruction->set_sharding(sharding);
  }

  if (proto.has_gather_dimension_numbers()) {
    instruction->gather_dimension_numbers_ =
        MakeUnique<GatherDimensionNumbers>(proto.gather_dimension_numbers());
  }
  for (int64 bound : proto.gather_window_bounds()) {
    instruction->gather_window_bounds_.push_back(bound);
  }

  instruction->channel_name_ = proto.channel_name();
  instruction->cost_estimate_ns_ = proto.cost_estimate_ns();

  return std::move(instruction);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64 parameter_number, const Shape& shape, const string& name) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kParameter, shape));
  instruction->parameter_number_ = parameter_number;
  instruction->name_ = name;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTrace(
    const string& tag, HloInstruction* operand) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kTrace, ShapeUtil::MakeNil()));
  instruction->operands_.push_back(operand);
  instruction->literal_ = Literal::CreateR1U8(tag);
  operand->set_tracing(instruction.get());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    std::unique_ptr<Literal> literal) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConstant, literal->shape()));
  instruction->literal_ = std::move(literal);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetTupleElement(const Shape& shape,
                                      HloInstruction* operand, int64 index) {
  CHECK(ShapeUtil::IsTuple(operand->shape()));
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kGetTupleElement, shape));
  instruction->tuple_index_ = index;
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRng(
    const Shape& shape, RandomDistribution distribution,
    tensorflow::gtl::ArraySlice<HloInstruction*> parameters) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kRng, shape));
  instruction->distribution_ = distribution;
  instruction->shape_ = shape;
  for (HloInstruction* param : parameters) {
    instruction->AppendOperand(param);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!ShapeUtil::IsOpaque(shape));
  }
  auto instruction = WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kClz:
    case HloOpcode::kDomain:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
      break;
    default:
      LOG(FATAL) << "Invalid unary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {operand});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kComplex:
    case HloOpcode::kDot:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTernary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs, HloInstruction* ehs) {
  // Only certain opcodes are supported with CreateTernary: opcodes of ternary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case (HloOpcode::kClamp):
    case (HloOpcode::kSelect):
      break;
    default:
      LOG(FATAL) << "Invalid ternary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs, ehs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateVariadic(
    const Shape& shape, HloOpcode opcode,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  CHECK_EQ(HloOpcode::kTuple, opcode);
  return CreateNary(shape, opcode, operands);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateMap(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* map_computation,
    tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  CHECK(static_operands.empty()) << "static_operands not yet supported";
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kMap, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->called_computations_.push_back(map_computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvolve(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConvolution, shape));
  if (window_util::HasBaseDilation(window)) {
    instruction->name_ = instruction->name() + "-base-dilated";
  }
  if (window_util::HasWindowDilation(window)) {
    instruction->name_ = instruction->name() + "-window-dilated";
  }
  instruction->AppendOperand(lhs);
  instruction->AppendOperand(rhs);
  instruction->window_ = MakeUnique<Window>(window);
  instruction->convolution_dimension_numbers_ =
      MakeUnique<ConvolutionDimensionNumbers>(dimension_numbers);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFft(
    const Shape& shape, HloInstruction* operand, FftType fft_type,
    tensorflow::gtl::ArraySlice<int64> fft_length) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kFft, shape));
  instruction->AppendOperand(operand);
  instruction->fft_type_ = fft_type;
  instruction->fft_length_.assign(fft_length.begin(), fft_length.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kDot, shape));
  instruction->AppendOperand(lhs);
  instruction->AppendOperand(rhs);
  instruction->dot_dimension_numbers_ =
      MakeUnique<DotDimensionNumbers>(dimension_numbers);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCanonicalDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs) {
  CHECK_EQ(ShapeUtil::Rank(lhs->shape()), 2);
  CHECK_EQ(ShapeUtil::Rank(rhs->shape()), 2);

  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kDot, shape));
  instruction->AppendOperand(lhs);
  instruction->AppendOperand(rhs);
  instruction->dot_dimension_numbers_ = MakeUnique<DotDimensionNumbers>();
  instruction->dot_dimension_numbers_->add_lhs_contracting_dimensions(1);
  instruction->dot_dimension_numbers_->add_rhs_contracting_dimensions(0);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateReducePrecision(const Shape& shape,
                                      HloInstruction* operand,
                                      const int exponent_bits,
                                      const int mantissa_bits) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kReducePrecision, shape));
  instruction->AppendOperand(operand);
  instruction->exponent_bits_ = exponent_bits;
  instruction->mantissa_bits_ = mantissa_bits;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCrossReplicaSum(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return CreateNary(shape, HloOpcode::kCrossReplicaSum, operands);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateInfeed(
    const Shape& shape, const string& config) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kInfeed, shape));
  instruction->set_infeed_config(config);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateOutfeed(
    const Shape& shape, HloInstruction* operand,
    tensorflow::StringPiece outfeed_config) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeNil()));
  CHECK(ShapeUtil::Compatible(operand->shape(), shape))
      << "Outfeed shape " << shape << " must be compatible with operand shape "
      << operand->shape();
  instruction->AppendOperand(operand);
  instruction->outfeed_config_ = std::string(outfeed_config);
  instruction->outfeed_shape_ = shape;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSend(
    HloInstruction* operand, int64 channel_id) {
  // Send instruction produces a tuple of {aliased operand, U32 context}.
  Shape output_shape = ShapeUtil::MakeTupleShape(
      {operand->shape(), ShapeUtil::MakeShape(U32, {})});
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kSend, output_shape));
  instruction->AppendOperand(operand);
  instruction->channel_id_ = channel_id;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSendDone(
    HloInstruction* operand) {
  CHECK(operand->opcode() == HloOpcode::kSend)
      << "SendDone must take the context operand from Send";
  auto instruction = WrapUnique(
      new HloInstruction(HloOpcode::kSendDone, ShapeUtil::MakeNil()));
  instruction->AppendOperand(operand);
  instruction->channel_id_ = operand->channel_id();
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecv(
    const Shape& shape, int64 channel_id) {
  // Recv instruction produces a tuple of {receive buffer, U32 context}.
  Shape output_shape =
      ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})});
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kRecv, output_shape));
  instruction->channel_id_ = channel_id;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecvDone(
    HloInstruction* operand) {
  CHECK(operand->opcode() == HloOpcode::kRecv)
      << "RecvDone must take the context operand from Recv";
  Shape output_shape = ShapeUtil::GetTupleElementShape(operand->shape(), 0);
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kRecvDone, output_shape));
  instruction->AppendOperand(operand);
  instruction->channel_id_ = operand->channel_id();
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReverse(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReverse, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(dimensions.begin(), dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateWhile(
    const Shape& shape, HloComputation* condition, HloComputation* body,
    HloInstruction* init) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kWhile, shape));
  instruction->AppendOperand(init);
  // Body comes before condition computation in the vector.
  instruction->called_computations_.push_back(body);
  instruction->called_computations_.push_back(condition);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* pred,
    HloInstruction* true_computation_arg, HloComputation* true_computation,
    HloInstruction* false_computation_arg, HloComputation* false_computation) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(pred);
  instruction->AppendOperand(true_computation_arg);
  instruction->AppendOperand(false_computation_arg);
  // In called_computations_, the index of true_computation must be 0 and that
  // of false computation must be 1, as defined by kTrueComputationIndex and
  // kFalseComputationIndex.
  instruction->called_computations_.push_back(true_computation);
  instruction->called_computations_.push_back(false_computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSlice(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kSlice, shape));
  instruction->AppendOperand(operand);
  instruction->slice_starts_.assign(start_indices.begin(), start_indices.end());
  instruction->slice_limits_.assign(limit_indices.begin(), limit_indices.end());
  instruction->slice_strides_.assign(strides.begin(), strides.end());
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (instruction->slice_strides_.empty()) {
    instruction->slice_strides_ = std::vector<int64>(start_indices.size(), 1LL);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicSlice(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kDynamicSlice, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(start_indices);
  instruction->dynamic_slice_sizes_.assign(slice_sizes.begin(),
                                           slice_sizes.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateDynamicUpdateSlice(const Shape& shape,
                                         HloInstruction* operand,
                                         HloInstruction* update,
                                         HloInstruction* start_indices) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kDynamicUpdateSlice, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(update);
  instruction->AppendOperand(start_indices);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConcatenate(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    int64 dimension) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConcatenate, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->dimensions_.push_back(dimension);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvert(
    const Shape& shape, HloInstruction* operand) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBitcastConvert(const Shape& shape,
                                     HloInstruction* operand) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBitcastConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReduce, shape));
  instruction->AppendOperand(arg);
  instruction->AppendOperand(init_value);
  instruction->dimensions_.assign(dimensions_to_reduce.begin(),
                                  dimensions_to_reduce.end());
  instruction->called_computations_.push_back(reduce_computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kReduceWindow, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(init_value);
  instruction->called_computations_.push_back(reduce_computation);
  instruction->window_ = MakeUnique<Window>(window);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormTraining(const Shape& shape,
                                        HloInstruction* operand,
                                        HloInstruction* scale,
                                        HloInstruction* offset, float epsilon,
                                        int64 feature_index) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBatchNormTraining, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(scale);
  instruction->AppendOperand(offset);
  instruction->epsilon_ = epsilon;
  instruction->feature_index_ = feature_index;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormInference(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64 feature_index) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBatchNormInference, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(scale);
  instruction->AppendOperand(offset);
  instruction->AppendOperand(mean);
  instruction->AppendOperand(variance);
  instruction->epsilon_ = epsilon;
  instruction->feature_index_ = feature_index;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormGrad(const Shape& shape, HloInstruction* operand,
                                    HloInstruction* scale, HloInstruction* mean,
                                    HloInstruction* variance,
                                    HloInstruction* grad_output, float epsilon,
                                    int64 feature_index) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBatchNormGrad, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(scale);
  instruction->AppendOperand(mean);
  instruction->AppendOperand(variance);
  instruction->AppendOperand(grad_output);
  instruction->epsilon_ = epsilon;
  instruction->feature_index_ = feature_index;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSelectAndScatter(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kSelectAndScatter, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(source);
  instruction->AppendOperand(init_value);
  // Select comes before scatter in the vector.
  instruction->called_computations_.push_back(select);
  instruction->called_computations_.push_back(scatter);
  instruction->window_ = MakeUnique<Window>(window);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBroadcast(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBroadcast, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(broadcast_dimensions.begin(),
                                  broadcast_dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBroadcastSequence(
    const Shape& output_shape, HloInstruction* operand,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        adder) {
  CHECK(ShapeUtil::IsScalar(operand->shape()) ||
        ShapeUtil::Rank(operand->shape()) == ShapeUtil::Rank(output_shape));
  Shape broadcast_shape = ShapeUtil::ChangeElementType(
      output_shape, operand->shape().element_type());
  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(operand->shape())) {
    auto broadcast =
        HloInstruction::CreateBroadcast(broadcast_shape, operand, {});
    broadcast->set_metadata(operand->metadata());
    if (operand->has_sharding()) {
      broadcast->set_sharding(operand->sharding());
    }
    return broadcast;
  }
  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64> broadcast_dimensions;
  std::vector<int64> reshaped_dimensions;
  for (int i = 0; i < ShapeUtil::Rank(operand->shape()); i++) {
    if (operand->shape().dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand->shape().dimensions(i));
    } else {
      CHECK_EQ(operand->shape().dimensions(i), 1)
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand: "
          << operand->ToString() << "; output_shape: " << output_shape;
    }
  }
  // Eliminate the size one dimensions.
  HloInstruction* reshaped_operand = adder(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(),
                           reshaped_dimensions),
      operand));
  reshaped_operand->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    reshaped_operand->set_sharding(operand->sharding());
  }
  // Broadcast 'reshape' up to the larger size.
  auto broadcast = HloInstruction::CreateBroadcast(
      broadcast_shape, reshaped_operand, broadcast_dimensions);
  broadcast->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    broadcast->set_sharding(operand->sharding());
  }
  return broadcast;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreatePad(
    const Shape& shape, HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kPad, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(padding_value);
  instruction->padding_config_ = MakeUnique<PaddingConfig>(padding_config);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReshape(
    const Shape& shape, HloInstruction* operand) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape),
           ShapeUtil::ElementsIn(operand->shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(operand->shape());
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReshape, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTranspose(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  CHECK_EQ(shape.dimensions().size(), dimensions.size());
  CHECK_EQ(shape.dimensions().size(), operand->shape().dimensions().size());
  CHECK(std::equal(operand->shape().dimensions().begin(),
                   operand->shape().dimensions().end(),
                   Permute(dimensions, shape.dimensions()).begin()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << ", operand->shape(): " << ShapeUtil::HumanString(shape)
      << ", dimensions: {" << Join(dimensions, ", ") << "}";
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kTranspose, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(dimensions.begin(), dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kFusion, shape));
  instruction->fusion_kind_ = fusion_kind;
  instruction->name_ = "fusion";
  instruction->set_parent(fused_root->parent());
  instruction->set_metadata(fused_root->metadata());
  instruction->CloneAndFuseInternal(fused_root);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* fusion_computation) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kFusion, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->fusion_kind_ = fusion_kind;
  instruction->name_ = "fusion";
  instruction->called_computations_.push_back(fusion_computation);
  fusion_computation->SetFusionInstruction(instruction.get());
  return instruction;
}

void HloInstruction::set_device_sharding(int64 device) {
  HloSharding device_sharding = HloSharding::AssignDevice(device);
  if (ShapeUtil::IsTuple(shape())) {
    set_sharding(HloSharding::Tuple(device_sharding.GetAsShapeTree(shape())));
  } else {
    set_sharding(device_sharding);
  }
}

void HloInstruction::SetupDerivedInstruction(
    HloInstruction* derived_instruction) const {
  if (sharding_ != nullptr) {
    derived_instruction->set_sharding(*sharding_);
  } else {
    derived_instruction->clear_sharding();
  }
  derived_instruction->set_metadata(metadata_);
}

HloInstruction* HloInstruction::AddFusionOperand(HloInstruction* new_operand) {
  CHECK_EQ(opcode(), HloOpcode::kFusion);
  CHECK_EQ(operand_count(),
           fused_instructions_computation()->parameter_instructions().size());
  const int64 param_no = operand_count();
  // Name the parameter after the instruction it represents in the outer
  // (non-fusion) computation.
  string param_name = StrCat(new_operand->name(), ".param_", param_no);
  HloInstruction* fused_parameter =
      fused_instructions_computation()->AddParameter(
          HloInstruction::CreateParameter(param_no, new_operand->shape(),
                                          param_name));
  AppendOperand(new_operand);
  return fused_parameter;
}

void HloInstruction::MergeFusionInstruction(
    HloInstruction* instruction_to_merge) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK_EQ(instruction_to_merge->opcode(), HloOpcode::kFusion);
  CHECK(std::find(operands().begin(), operands().end(), instruction_to_merge) !=
        operands().end());
  // Clone the instruction from which to merge fused instructions.
  std::unique_ptr<HloInstruction> clone = instruction_to_merge->Clone();
  // Replace uses of fused parameters with the corresponding operand of the
  // fusion.  Add all non-parameter fused instructions to 'unfused_instructions'
  // to be merged into 'this'.  This is done in reverse post order.
  std::vector<HloInstruction*> unfused_instructions;
  auto fused_instructions =
      clone->fused_instructions_computation()->MakeInstructionPostOrder();
  for (auto fused_it = fused_instructions.rbegin();
       fused_it != fused_instructions.rend(); ++fused_it) {
    auto fused_instruction = *fused_it;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      TF_CHECK_OK(fused_instruction->ReplaceAllUsesWith(
          clone->mutable_operand(fused_instruction->parameter_number())));
    } else {
      unfused_instructions.push_back(fused_instruction);
    }
  }
  CHECK(unfused_instructions.front() == clone->fused_expression_root());
  // Replace instruction_to_merge use of 'this' with unfused_root.
  TF_CHECK_OK(
      instruction_to_merge->ReplaceUseWith(this, unfused_instructions.front()));
  // Fuse 'unfused_instructions' into 'this'.
  for (auto& instruction : unfused_instructions) {
    FuseInstruction(instruction);
    instruction->DetachFromOperands();
  }
  CHECK_EQ(0, clone->user_count());
  clone->DetachFromOperands();
  TF_CHECK_OK(parent()->parent()->RemoveEmbeddedComputation(
      clone->fused_instructions_computation()));
}

void HloInstruction::MergeFusionInstructionIntoMultiOutput(
    HloInstruction* instruction_to_merge) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK_EQ(instruction_to_merge->opcode(), HloOpcode::kFusion);
  // Add all non-parameter fused instructions to 'unfused_instructions' to be
  // merged into 'this'. `old_to_new' maps the instructions in the fused node
  // to the disaseembled fusion instructions.
  // Note that we add the unfused instructions to this->parent_ computation.
  // This is necessary because the unique_id needs for an instruction and
  // it's only added when inserting to the computation.
  tensorflow::gtl::FlatMap<HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> unfused_instructions;
  auto computation_to_merge =
      instruction_to_merge->fused_instructions_computation();
  auto post_order = computation_to_merge->MakeInstructionPostOrder();
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto fused_instruction = *rit;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&old_to_new, fused_instruction,
                  instruction_to_merge->mutable_operand(
                      fused_instruction->parameter_number()));
      continue;
    }

    // Here we clone the insertion and call FuseInstructionIntoMultiOutput()
    // which clones again. This can be improved.
    auto cloned_instruction =
        parent_->AddInstruction(fused_instruction->Clone());
    unfused_instructions.push_back(cloned_instruction);
    InsertOrDie(&old_to_new, fused_instruction, cloned_instruction);
  }
  for (auto unfused_instruction : unfused_instructions) {
    for (int64 index = 0; index < unfused_instruction->operand_count();
         index++) {
      auto new_operand =
          FindOrDie(old_to_new, unfused_instruction->mutable_operand(index));
      TF_CHECK_OK(unfused_instruction->ReplaceOperandWith(index, new_operand));
    }
  }

  HloInstruction* unfused_root = unfused_instructions.front();
  TF_CHECK_OK(instruction_to_merge->ReplaceAllUsesWith(unfused_root));

  TF_CHECK_OK(
      instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge));
  if (GetModule()) {
    TF_CHECK_OK(GetModule()->RemoveEmbeddedComputation(computation_to_merge));
  }

  // Fuse the root instruction and generate multiple outputs.
  FuseInstructionIntoMultiOutput(unfused_root);
  TF_CHECK_OK(unfused_root->parent()->RemoveInstruction(unfused_root));
  // The rest instructions are of normal fusing.
  for (int64 i = 1; i < unfused_instructions.size(); i++) {
    auto instruction = unfused_instructions[i];
    FuseInstruction(instruction);
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
}

HloInstruction* HloInstruction::FuseInstructionInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);

  // When add_output is false, this fusion instruction must be a user of
  // instruction_to_fuse.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_fuse));
  }
  HloInstruction* fused_instruction =
      CloneAndFuseInternal(instruction_to_fuse, add_output);
  return fused_instruction;
}

HloInstruction* HloInstruction::CloneAndFuseInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK(instruction_to_fuse->IsFusable()) << instruction_to_fuse->ToString();
  VLOG(3) << "CloneAndFuseInternal:\n" << instruction_to_fuse->ToString();
  HloInstruction* clone = nullptr;
  if (called_computations_.empty()) {
    // New fusion instruction. It should not be a multioutput instruction.
    CHECK(!add_output);
    auto builder = HloComputation::Builder("fused_computation", this);
    builder.AddInstruction(instruction_to_fuse->Clone(/*suffix=*/""));
    called_computations_.push_back(
        CHECK_NOTNULL(GetModule())->AddEmbeddedComputation(builder.Build()));
    clone = fused_expression_root();
  } else {
    clone = fused_instructions_computation()->AddInstruction(
        instruction_to_fuse->Clone(/*suffix=*/""));
    // When add_output is false, instruction_to_fuse is necessarily an operand
    // of the fusion instruction. After fusion this will no longer be the case.
    // Remove the operand from the operand list and remove its corresponding
    // fused parameter instruction. Renumber parameters as necessary to make
    // parameter numbers consistent with their index in the
    // fused_parameter_ vector.
    bool in_operand_list = std::find(operands_.begin(), operands_.end(),
                                     instruction_to_fuse) != operands_.end();
    CHECK(add_output || in_operand_list);
    const std::vector<HloInstruction*>& fused_parameters =
        fused_instructions_computation()->parameter_instructions();
    for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
      if (instruction_to_fuse == operands_[operand_num]) {
        // replace the fused parameter instruction's uses with the clone.
        HloInstruction* fused_parameter = fused_parameters[operand_num];
        TF_CHECK_OK(fused_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding fused parameter and operand from their
        // respective vectors.
        TF_CHECK_OK(
            fused_instructions_computation()->RemoveParameter(operand_num));
        operands_.erase(operands_.begin() + operand_num);
        break;
      }
    }
    // We've cloned instruction_to_fuse into this fusion instruction, so this
    // fusion instruction is no longer a use of instruction_to_fuse.
    if (in_operand_list) {
      instruction_to_fuse->RemoveUser(this);
      // When the instruction_to_fuse does not have other users, we don't need
      // to generate a multioutput fusion instruction.
      if (instruction_to_fuse->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const std::vector<HloInstruction*>& fused_parameters =
      fused_instructions_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the fusion instruction. A
  // complication is that some clone operands may already be operands of the
  // fusion instruction.
  for (int64 operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the fusion node.
    CHECK_EQ(operands_.size(), fused_parameters.size());
    HloInstruction* fused_param = nullptr;
    for (int64 i = 0; i < operands_.size(); ++i) {
      if (operands_[i] == operand) {
        fused_param = fused_parameters[i];
        break;
      }
    }

    if (fused_param == nullptr) {
      // Clone's operand was not already an operand of the fusion
      // instruction. Add it as an operand and add a corresponding fused
      // parameter instruction.
      fused_param = AddFusionOperand(operand);
    }
    TF_CHECK_OK(clone->ReplaceOperandWith(operand_num, fused_param));
  }

  if (add_output) {
    CHECK_GT(instruction_to_fuse->user_count(), 0);
    // If this is already a multioutput fusion instruction, expand the root
    // tuple by 1.
    HloInstruction* fused_root = fused_expression_root();
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (fused_root->opcode() == HloOpcode::kTuple) {
      tuple_elements = fused_root->operands();
    } else {
      tuple_elements.push_back(fused_root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction* new_root = fused_instructions_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_instructions_computation()->set_root_instruction(new_root);
    shape_ = new_root->shape();
    if (fused_root->opcode() == HloOpcode::kTuple) {
      TF_CHECK_OK(
          fused_instructions_computation()->RemoveInstruction(fused_root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original fusion instruction.
    if (newly_created_tuple_instr) {
      HloInstruction* new_instr = parent_->AddInstruction(
          HloInstruction::CreateGetTupleElement(fused_root->shape(), this, 0));
      TF_CHECK_OK(ReplaceAllUsesWith(new_instr));
    }
    int64 index = tuple_elements.size();
    if (instruction_to_fuse->opcode() == HloOpcode::kTuple) {
      index -= instruction_to_fuse->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      for (auto old_gte : instruction_to_fuse->users()) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64 old_tuple_index = old_gte->tuple_index();
        HloInstruction* new_gte =
            parent_->AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        TF_CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        TF_CHECK_OK(parent_->RemoveInstruction(old_gte));
      }
      TF_CHECK_OK(fused_instructions_computation()->RemoveInstruction(clone));
    } else {
      HloInstruction* new_gte =
          parent_->AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      TF_CHECK_OK(instruction_to_fuse->ReplaceAllUsesWith(new_gte));
    }
  }

  VLOG(2) << "New clone:\n" << clone->ToString();
  return clone;
}

RandomDistribution HloInstruction::random_distribution() const {
  CHECK_EQ(opcode_, HloOpcode::kRng);
  return distribution_;
}

bool HloInstruction::HasSideEffectNoRecurse() const {
  switch (opcode_) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kRng:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
    case HloOpcode::kHostCompute:
      return true;
    default:
      return false;
  }
}

bool HloInstruction::HasSideEffect() const {
  if (HasSideEffectNoRecurse()) {
    return true;
  }
  // Check if any of the called computations has a side effect.
  for (const auto& computation : called_computations()) {
    if (computation->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* computation) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->called_computations_.push_back(computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kCustomCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->custom_call_target_ = std::string(custom_call_target);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateHostCompute(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece channel_name, const int64 cost_estimate_ns) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kHostCompute, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->channel_name_ = std::string(channel_name);
  instruction->cost_estimate_ns_ = cost_estimate_ns;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTuple(
    tensorflow::gtl::ArraySlice<HloInstruction*> elements) {
  std::vector<Shape> element_shapes;
  for (auto element : elements) {
    element_shapes.push_back(element->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);
  return CreateVariadic(tuple_shape, HloOpcode::kTuple, elements);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateGather(
    const Shape& shape, HloInstruction* operand, HloInstruction* gather_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    tensorflow::gtl::ArraySlice<int64> window_bounds) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kGather, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(gather_indices);
  instruction->gather_dimension_numbers_ =
      MakeUnique<GatherDimensionNumbers>(gather_dim_numbers);
  c_copy(window_bounds, std::back_inserter(instruction->gather_window_bounds_));
  return instruction;
}

/* static */ GatherDimensionNumbers HloInstruction::MakeGatherDimNumbers(
    tensorflow::gtl::ArraySlice<int64> output_window_dims,
    tensorflow::gtl::ArraySlice<int64> elided_window_dims,
    tensorflow::gtl::ArraySlice<int64> gather_dims_to_operand_dims,
    int64 index_vector_dim) {
  GatherDimensionNumbers gather_dim_numbers;
  for (int64 output_window_dim : output_window_dims) {
    gather_dim_numbers.add_output_window_dims(output_window_dim);
  }
  for (int64 elided_window_dim : elided_window_dims) {
    gather_dim_numbers.add_elided_window_dims(elided_window_dim);
  }
  for (int64 gather_dim_to_input_dim : gather_dims_to_operand_dims) {
    gather_dim_numbers.add_gather_dims_to_operand_dims(gather_dim_to_input_dim);
  }

  gather_dim_numbers.set_index_vector_dim(index_vector_dim);
  return gather_dim_numbers;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDomain(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kDomain, shape));
  instruction->operand_side_metadata_ = std::move(operand_side_metadata);
  instruction->user_side_metadata_ = std::move(user_side_metadata);
  instruction->AppendOperand(operand);
  return instruction;
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  VLOG(3) << "CloneWithNewOperands:\n  " << ToString();
  VLOG(3) << "  new operands:";
  for (const HloInstruction* new_operand : new_operands) {
    VLOG(3) << "    %" << new_operand->name();
  }

  std::unique_ptr<HloInstruction> clone;
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateUnary(shape, opcode_, new_operands[0]);
      break;
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateBinary(shape, opcode_, new_operands[0], new_operands[1]);
      break;
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateTernary(shape, opcode_, new_operands[0], new_operands[1],
                            new_operands[2]);
      break;
    // Other supported ops.
    case HloOpcode::kBroadcast:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateBroadcast(shape, new_operands[0], dimensions_);
      break;
    case HloOpcode::kCall:
      clone = CreateCall(shape, new_operands, to_apply());
      break;
    case HloOpcode::kCustomCall:
      clone = CreateCustomCall(shape, new_operands, custom_call_target_);
      if (window_ != nullptr) {
        clone->window_ = MakeUnique<Window>(*window_);
      }
      if (convolution_dimension_numbers_ != nullptr) {
        clone->convolution_dimension_numbers_ =
            MakeUnique<ConvolutionDimensionNumbers>(
                *convolution_dimension_numbers_);
      }
      break;
    case HloOpcode::kHostCompute:
      clone = CreateHostCompute(shape, new_operands, channel_name_,
                                cost_estimate_ns_);
      break;
    case HloOpcode::kConcatenate:
      clone = CreateConcatenate(shape, new_operands, dimensions(0));
      break;
    case HloOpcode::kConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kBitcastConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateBitcastConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kReducePrecision:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateReducePrecision(shape, new_operands[0], exponent_bits_,
                                    mantissa_bits_);
      break;
    case HloOpcode::kConvolution:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateConvolve(shape, new_operands[0], new_operands[1], *window_,
                             *convolution_dimension_numbers_);
      break;
    case HloOpcode::kDot:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateDot(shape, new_operands[0], new_operands[1],
                        *dot_dimension_numbers_);
      break;
    case HloOpcode::kFft:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateFft(shape, new_operands[0], fft_type_, fft_length_);
      break;
    case HloOpcode::kCrossReplicaSum:
      clone = CreateCrossReplicaSum(shape, new_operands);
      break;
    case HloOpcode::kGetTupleElement:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateGetTupleElement(shape, new_operands[0], tuple_index());
      break;
    case HloOpcode::kMap:
      clone = CreateMap(shape, new_operands, to_apply());
      break;
    case HloOpcode::kPad:
      CHECK_EQ(new_operands.size(), 2);
      clone =
          CreatePad(shape, new_operands[0], new_operands[1], *padding_config_);
      break;
    case HloOpcode::kReduce:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateReduce(shape, new_operands[0], new_operands[1], dimensions_,
                           to_apply());
      break;
    case HloOpcode::kReduceWindow:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateReduceWindow(shape, new_operands[0], new_operands[1],
                                 *window_, to_apply());
      break;
    case HloOpcode::kSelectAndScatter:
      CHECK_EQ(new_operands.size(), 3);
      clone =
          CreateSelectAndScatter(shape, new_operands[0], select(), *window_,
                                 new_operands[1], new_operands[2], scatter());
      break;
    case HloOpcode::kReverse:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateReverse(shape, new_operands[0], dimensions_);
      break;
    case HloOpcode::kRng:
      clone = CreateRng(shape, distribution_, new_operands);
      break;
    case HloOpcode::kReshape:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateReshape(shape, new_operands[0]);
      break;
    case HloOpcode::kSlice:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateSlice(shape, new_operands[0], slice_starts_, slice_limits_,
                          slice_strides_);
      break;
    case HloOpcode::kDynamicSlice:
      clone = CreateDynamicSlice(shape, new_operands[0], new_operands[1],
                                 dynamic_slice_sizes_);
      break;
    case HloOpcode::kDynamicUpdateSlice:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateDynamicUpdateSlice(shape, new_operands[0], new_operands[1],
                                       new_operands[2]);
      break;
    case HloOpcode::kTranspose:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateTranspose(shape, new_operands[0], dimensions_);
      break;
    case HloOpcode::kTuple:
      clone = CreateTuple(new_operands);
      *clone->mutable_shape() = shape;
      break;
    case HloOpcode::kWhile:
      CHECK_EQ(new_operands.size(), 1);
      clone =
          CreateWhile(shape, while_condition(), while_body(), new_operands[0]);
      break;
    case HloOpcode::kConstant:
      clone = CreateConstant(literal_->CloneToUnique());
      break;
    case HloOpcode::kFusion: {
      HloModule* module = context != nullptr ? context->module() : GetModule();
      HloComputation* new_fused_computation = nullptr;
      if (context != nullptr) {
        new_fused_computation =
            context->FindComputation(fused_instructions_computation());
      }
      if (new_fused_computation == nullptr) {
        new_fused_computation = module->AddEmbeddedComputation(
            fused_instructions_computation()->Clone("clone", context));
      }
      clone = CreateFusion(/*shape=*/shape, /*fusion_kind=*/fusion_kind(),
                           /*operands=*/new_operands,
                           /*fusion_computation=*/new_fused_computation);
      break;
    }
    case HloOpcode::kParameter:
      clone = CreateParameter(parameter_number_, shape, name_);
      break;
    case HloOpcode::kBatchNormTraining:
      CHECK_EQ(new_operands.size(), 3);
      clone =
          CreateBatchNormTraining(shape, new_operands[0], new_operands[1],
                                  new_operands[2], epsilon(), feature_index());
      break;
    case HloOpcode::kBatchNormInference:
      CHECK_EQ(new_operands.size(), 5);
      clone = CreateBatchNormInference(
          shape, new_operands[0], new_operands[1], new_operands[2],
          new_operands[3], new_operands[4], epsilon(), feature_index());
      break;
    case HloOpcode::kInfeed:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreateInfeed(shape, infeed_config());
      break;
    case HloOpcode::kOutfeed:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateOutfeed(outfeed_shape_, new_operands[0], outfeed_config());
      break;
    case HloOpcode::kBatchNormGrad:
      CHECK_EQ(new_operands.size(), 5);
      clone = CreateBatchNormGrad(shape, new_operands[0], new_operands[1],
                                  new_operands[2], new_operands[3],
                                  new_operands[4], epsilon(), feature_index());
      break;
    case HloOpcode::kConditional:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateConditional(shape, new_operands[0], new_operands[1],
                                true_computation(), new_operands[2],
                                false_computation());
      break;
    case HloOpcode::kSend:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateSend(new_operands[0], channel_id());
      break;
    case HloOpcode::kSendDone:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateSendDone(new_operands[0]);
      break;
    case HloOpcode::kRecv:
      CHECK_EQ(new_operands.size(), 0);
      // The shape is a tuple, but CreateRecv() wants the raw data shape.
      clone =
          CreateRecv(ShapeUtil::GetTupleElementShape(shape, 0), channel_id());
      break;
    case HloOpcode::kRecvDone:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateRecvDone(new_operands[0]);
      break;
    case HloOpcode::kGather:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateGather(shape, new_operands[0], new_operands[1],
                           *gather_dimension_numbers_, gather_window_bounds_);
      break;
    case HloOpcode::kDomain:
      CHECK_EQ(new_operands.size(), 1);
      clone =
          CreateDomain(shape, new_operands[0], operand_side_metadata_->Clone(),
                       user_side_metadata_->Clone());
      break;
    case HloOpcode::kTrace:
      LOG(FATAL) << "Not yet implemented, clone: " << HloOpcodeString(opcode_);
  }
  SetupDerivedInstruction(clone.get());
  clone->set_parent(parent_);
  clone->set_raw_backend_config_string(backend_config_);
  if (context != nullptr) {
    context->MapInstruction(this, clone.get());
    clone->ReplaceCalledComputations([&](HloComputation* callee) {
      return callee->parent() != context->module()
                 ? context->module()->DeepCloneComputation(callee, context)
                 : callee;
    });
  }
  return clone;
}

HloInstruction::~HloInstruction() {}

std::unique_ptr<HloInstruction> HloInstruction::Clone(
    const string& suffix, HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape_, operands_, context);
  if (suffix.empty()) {
    clone->name_ = name();
  } else {
    // If an instruction is cloned multiple times avoid names like
    // foo.suffix.suffix.suffix. Instead of repeating the suffix add a numeric
    // suffix. Specifically, the clone of foo.suffix is named foo.suffix2, the
    // clone of foo.suffix2 is named foo.suffix3 and so on.
    const string dot_suffix = "." + suffix;
    size_t index = name().rfind(dot_suffix);
    if (index == string::npos) {
      // Existing name does not include ".suffix".
      clone->name_ = name() + dot_suffix;
    } else {
      // Existing name includes ".suffix". Determine if substring after
      // ".suffix" is numeric and should be replaced with an incremented number.
      string after_suffix = name().substr(index + dot_suffix.size());
      if (after_suffix.empty()) {
        // Existing name ends in ".suffix". New name should end in ".suffix2".
        clone->name_ = name() + "2";
      } else {
        // If names ends with .suffix[0-9]+ then replace with a suffix with the
        // numeric value incremented.
        int64 numeric_suffix;
        if (tensorflow::strings::safe_strto64(after_suffix, &numeric_suffix)) {
          clone->name_ =
              StrCat(name().substr(0, index), dot_suffix, numeric_suffix + 1);
        } else {
          // Substring after ".suffix" is non-numeric.
          clone->name_ = name() + dot_suffix;
        }
      }
    }
  }
  return clone;
}

std::pair<const HloInstruction*, ShapeIndex>
HloInstruction::LatestNonGteAncestorAndIndex() const {
  const HloInstruction* hlo = this;
  ShapeIndex index;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    index.push_back(hlo->tuple_index());
    hlo = hlo->operand(0);
  }

  // We built up index in the reverse order from what we want.
  std::reverse(index.begin(), index.end());

  return {hlo, index};
}

const HloInstruction* HloInstruction::LatestNonGteAncestor() const {
  const HloInstruction* hlo = this;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    hlo = hlo->operand(0);
  }
  return hlo;
}

const Literal& HloInstruction::literal() const {
  CHECK_EQ(HloOpcode::kConstant, opcode_);
  return *literal_;
}

bool HloInstruction::HasLiteral() const { return literal_ != nullptr; }

bool HloInstruction::CanHaveDimensionsField() const {
  return (opcode() == HloOpcode::kReverse ||
          opcode() == HloOpcode::kConcatenate ||
          opcode() == HloOpcode::kReduce || opcode() == HloOpcode::kBroadcast ||
          opcode() == HloOpcode::kTranspose);
}

const std::vector<int64>& HloInstruction::dimensions() const {
  CHECK(CanHaveDimensionsField());
  return dimensions_;
}

int64 HloInstruction::dimensions(int64 index) const {
  return dimensions()[index];
}

int64 HloInstruction::concatenate_dimension() const {
  CHECK(opcode() == HloOpcode::kConcatenate);
  CHECK_EQ(1, dimensions_.size());
  return dimensions(0);
}

int64 HloInstruction::tuple_index() const {
  CHECK_EQ(HloOpcode::kGetTupleElement, opcode_);
  return tuple_index_;
}

const HloInstruction* HloInstruction::operand(int64 i) const {
  return operands_[i];
}

HloInstruction* HloInstruction::mutable_operand(int64 i) {
  CHECK(operands_[i] != nullptr);
  return operands_[i];
}

int64 HloInstruction::operand_index(const HloInstruction* target) const {
  for (int64 i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand: " << target->ToString();
}

HloInstruction::InstructionVector HloInstruction::unique_operands() const {
  InstructionVector unique;
  tensorflow::gtl::FlatSet<const HloInstruction*> seen;
  for (HloInstruction* operand : operands()) {
    if (seen.insert(operand).second) {
      unique.push_back(operand);
    }
  }
  return unique;
}

Status HloInstruction::AddControlDependencyTo(HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (std::find(control_successors_.begin(), control_successors_.end(),
                instruction) == control_successors_.end()) {
    control_successors_.push_back(instruction);
    TF_RET_CHECK(std::find(instruction->control_predecessors_.begin(),
                           instruction->control_predecessors_.end(),
                           this) == instruction->control_predecessors_.end());
    instruction->control_predecessors_.push_back(this);
  }
  return Status::OK();
}

Status HloInstruction::RemoveControlDependencyTo(HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  TF_RETURN_IF_ERROR(EraseElementFromVector(&control_successors_, instruction));
  TF_RETURN_IF_ERROR(
      EraseElementFromVector(&instruction->control_predecessors_, this));
  return Status::OK();
}

Status HloInstruction::DropAllControlDeps() {
  for (auto* ctrl_succ : control_successors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_succ->control_predecessors_, this));
  }
  for (auto* ctrl_pred : control_predecessors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_pred->control_successors_, this));
  }
  control_successors_.clear();
  control_predecessors_.clear();
  return Status::OK();
}

Status HloInstruction::CopyAllControlDepsFrom(const HloInstruction* inst) {
  for (auto* ctrl_pred : inst->control_predecessors()) {
    TF_RETURN_IF_ERROR(ctrl_pred->AddControlDependencyTo(this));
  }

  for (auto* ctrl_succ : inst->control_successors()) {
    TF_RETURN_IF_ERROR(this->AddControlDependencyTo(ctrl_succ));
  }

  return Status::OK();
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::AddUser(HloInstruction* user) {
  if (!ContainsKey(user_set_, user)) {
    user_set_.insert(user);
    users_.push_back(user);
  }
}

bool HloInstruction::IsConstant() const {
  return opcode_ == HloOpcode::kConstant;
}

bool HloInstruction::HasConstantOperand() const {
  for (const HloInstruction* operand : operands_) {
    if (operand->IsConstant()) {
      return true;
    }
  }
  return false;
}

bool HloInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAbs:
    case HloOpcode::kAtan2:
    case HloOpcode::kAdd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kEq:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLe:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kAnd:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kNegate:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTuple:
      return true;

    // Broadcast, Concatenate, and Transpose need the same dimensions field.
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kTranspose:
      return dimensions() == other.dimensions();

    case HloOpcode::kFusion:
      return fusion_kind() == other.fusion_kind() &&
             eq_computations(fused_instructions_computation(),
                             other.fused_instructions_computation());

    // These opcodes have complex or special behavior so just return false.
    case HloOpcode::kDomain:
    case HloOpcode::kRng:
    case HloOpcode::kTrace:
    case HloOpcode::kWhile:
      return false;

    case HloOpcode::kParameter:
      return parameter_number() == other.parameter_number();

    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
      return feature_index() == other.feature_index() &&
             epsilon() == other.epsilon();

    // A constant is defined by the value in the literal.
    case HloOpcode::kConstant:
      return literal() == other.literal();

    // A reduce-precision operation is determined by the bit sizes.
    case HloOpcode::kReducePrecision:
      return exponent_bits() == other.exponent_bits() &&
             mantissa_bits() == other.mantissa_bits();

    // Convolution has a window and dimensions.
    case HloOpcode::kConvolution:
      return protobuf_util::ProtobufEquals(window(), other.window()) &&
             protobuf_util::ProtobufEquals(
                 convolution_dimension_numbers(),
                 other.convolution_dimension_numbers());
    // Check dot dimension numbers.
    case HloOpcode::kDot:
      return protobuf_util::ProtobufEquals(dot_dimension_numbers(),
                                           other.dot_dimension_numbers());

    case HloOpcode::kGather:
      return protobuf_util::ProtobufEquals(gather_dimension_numbers(),
                                           other.gather_dimension_numbers()) &&
             gather_window_bounds() == other.gather_window_bounds();

    // FFT has various types & lengths.
    case HloOpcode::kFft:
      return fft_type() == other.fft_type() &&
             fft_length() == other.fft_length();

    // Reduction results are determined by the reduction dimension and the
    // reduction computation.
    case HloOpcode::kReduce:
      return dimensions() == other.dimensions() &&
             eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kReduceWindow:
      return eq_computations(to_apply(), other.to_apply()) &&
             protobuf_util::ProtobufEquals(window(), other.window());

    // SelectAndScatter is determined by both select and scatter
    // computation as well as the window configuration.
    case HloOpcode::kSelectAndScatter:
      return eq_computations(select(), other.select()) &&
             eq_computations(scatter(), other.scatter()) &&
             protobuf_util::ProtobufEquals(window(), other.window());


    // Remaining instructions with special values.
    case HloOpcode::kGetTupleElement:
      return tuple_index() == other.tuple_index();
    case HloOpcode::kPad:
      return protobuf_util::ProtobufEquals(padding_config(),
                                           other.padding_config());
    case HloOpcode::kSlice:
      return slice_starts_ == other.slice_starts_ &&
             slice_limits_ == other.slice_limits_ &&
             slice_strides_ == other.slice_strides_;
    case HloOpcode::kCall:
    case HloOpcode::kMap:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kCustomCall:
      if ((window_ == nullptr) != (other.window_ == nullptr) ||
          (window_ != nullptr &&
           !protobuf_util::ProtobufEquals(window(), other.window()))) {
        return false;
      }
      if ((convolution_dimension_numbers_ == nullptr) !=
              (other.convolution_dimension_numbers_ == nullptr) ||
          (convolution_dimension_numbers_ != nullptr &&
           !protobuf_util::ProtobufEquals(
               convolution_dimension_numbers(),
               other.convolution_dimension_numbers()))) {
        return false;
      }
      return custom_call_target_ == other.custom_call_target_;
    case HloOpcode::kReverse:
      return dimensions() == other.dimensions();
    case HloOpcode::kConditional:
      return eq_computations(true_computation(), other.true_computation()) &&
             eq_computations(false_computation(), other.false_computation());

    // These opcodes are not yet supported.
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kSort:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kHostCompute:
      return false;
  }
}

bool HloInstruction::IsRank2Transpose() const {
  return (opcode_ == HloOpcode::kTranspose) &&
         dimensions_ == std::vector<int64>({1, 0}) &&
         shape_.dimensions_size() == 2 &&
         std::equal(shape_.dimensions().begin(), shape_.dimensions().end(),
                    operands_[0]->shape_.dimensions().rbegin());
}

void HloInstruction::RemoveUser(HloInstruction* user) {
  auto set_it = user_set_.find(user);
  CHECK(set_it != user_set_.end());
  user_set_.erase(set_it);
  // This is linear in the number of the users, but a vector provides a stable
  // iteration order and much faster traversal.
  auto vec_it = std::find(users_.begin(), users_.end(), user);
  CHECK(vec_it != users_.end());
  users_.erase(vec_it);
}

Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                      HloInstruction* new_producer) {
  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());

  VLOG(3) << "Replacing uses of " << name() << " in " << user->name()
          << " with " << new_producer->name();

  RemoveUser(user);

  TF_RET_CHECK(
      std::count(user->operands_.begin(), user->operands_.end(), this) >= 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
  return Status::OK();
}

Status HloInstruction::ReplaceOperandWith(int64 operand_num,
                                          HloInstruction* new_operand) {
  TF_RET_CHECK(operand_num >= 0);
  TF_RET_CHECK(operand_num < operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  TF_RET_CHECK(ShapeUtil::CompatibleIgnoringFpPrecision(old_operand->shape(),
                                                        new_operand->shape()))
      << old_operand->shape().ShortDebugString() << " is not compatible with "
      << new_operand->shape().ShortDebugString();
  operands_[operand_num] = new_operand;

  VLOG(3) << "Replacing operand " << operand_num << " of " << name() << " with "
          << new_operand->name() << ", was " << old_operand->name();

  if (std::find(operands_.begin(), operands_.end(), old_operand) ==
      operands_.end()) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
  return Status::OK();
}

Status HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer) {
  bool new_producer_is_user = false;
  for (HloInstruction* user : users()) {
    if (user == new_producer) {
      // It's possible that new_producer is a user of this instruction as might
      // be the case when replacing an instruction with a kCopy of itself. In
      // this case, don't do the replacement to avoid creating a cycle in the
      // graph. new_producer remains the only user of this instruction.
      new_producer_is_user = true;
    } else {
      std::replace(user->operands_.begin(), user->operands_.end(), this,
                   new_producer);
      new_producer->AddUser(user);
    }
  }
  users_.clear();
  user_set_.clear();
  if (new_producer_is_user) {
    AddUser(new_producer);
  }
  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer);
  }

  return Status::OK();
}

void HloInstruction::DetachFromOperands() {
  VLOG(3) << "DetachFromOperands:\n  " << ToString();
  CHECK_EQ(0, user_count());
  // An instruction may be repeated as an operand. To avoid calling RemoveUser
  // twice on the same operand, keep a set of already detached operands.
  std::set<HloInstruction*> detached_operands;
  for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (!ContainsKey(detached_operands, operand)) {
      operand->RemoveUser(this);
      detached_operands.insert(operand);
    }
    operands_[operand_num] = nullptr;
  }
}

HloComputation* HloInstruction::to_apply() const {
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
      CHECK_EQ(called_computations_.size(), 1);
      return called_computations_[0];
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

void HloInstruction::set_to_apply(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
      CHECK_EQ(called_computations_.size(), 1);
      called_computations_[0] = computation;
      break;
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

const string& HloInstruction::custom_call_target() const {
  CHECK_EQ(opcode_, HloOpcode::kCustomCall);
  return custom_call_target_;
}

const string& HloInstruction::outfeed_config() const {
  CHECK_EQ(opcode_, HloOpcode::kOutfeed);
  return outfeed_config_;
}

HloComputation* HloInstruction::while_condition() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kConditionComputationIndex];
}

HloComputation* HloInstruction::while_body() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kBodyComputationIndex];
}

void HloInstruction::set_while_condition(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kConditionComputationIndex] = computation;
}

void HloInstruction::set_while_body(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kBodyComputationIndex] = computation;
}

HloComputation* HloInstruction::select() const {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  return called_computations_[kSelectComputationIndex];
}

HloComputation* HloInstruction::scatter() const {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  return called_computations_[kScatterComputationIndex];
}

void HloInstruction::set_select(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  called_computations_[kSelectComputationIndex] = computation;
}

void HloInstruction::set_scatter(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  called_computations_[kScatterComputationIndex] = computation;
}

HloComputation* HloInstruction::true_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  return called_computations_[kTrueComputationIndex];
}

HloComputation* HloInstruction::false_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  return called_computations_[kFalseComputationIndex];
}

void HloInstruction::set_true_computation(HloComputation* true_computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  called_computations_[kTrueComputationIndex] = true_computation;
}

void HloInstruction::set_false_computation(HloComputation* false_computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  called_computations_[kFalseComputationIndex] = false_computation;
}

string HloInstruction::SignatureString() const {
  string operands =
      Join(operands_, ", ", [](string* out, HloInstruction* operand) {
        StrAppend(out, ShapeUtil::HumanString(operand->shape()));
      });
  return StrCat("(", operands, ") -> ", ShapeUtil::HumanString(shape()));
}

namespace {

string PrintName(const string& name, const HloPrintOptions& options) {
  return StrCat(options.print_percent() ? "%" : "", name);
}

}  // namespace

string HloInstruction::ToString(const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  return ToStringWithCanonicalNameMap(options, &new_map);
}

string HloInstruction::ToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string result = "";

  // Logic to print the instruction name (e.g. "%foo = ").
  if (options.canonicalize_instruction_names()) {
    if (options.is_in_nested_computation()) {
      // If we are canonicalizing instruction names and this is a top-level
      // HloInstruction::ToString() call, don't print an instruction name.
      StrAppend(&result,
                PrintName(canonical_name_map->LookupOrInsert(name()), options),
                " = ");
    }
  } else {
    StrAppend(&result, PrintName(name(), options), " = ");
  }

  // Print opcode, operand(s) and shape.
  StrAppend(&result, ShapeUtil::HumanStringWithLayout(shape()), " ",
            HloOpcodeString(opcode()), "(",
            OperandsToStringWithCanonicalNameMap(options, canonical_name_map),
            ")");

  // Print additional attributes. If an instruction contains a subcomputation,
  // the subcomputation is also printed here.
  for (const string& extra : ExtraAttributesToString(options)) {
    StrAppend(&result, ", ", extra);
  }

  if (options.print_metadata() &&
      (!metadata_.op_type().empty() || !metadata_.op_name().empty() ||
       !metadata_.source_file().empty())) {
    StrAppend(&result, ", metadata={", xla::OpMetadataToString(metadata_), "}");
  }
  if (options.print_backend_config() && !backend_config_.empty()) {
    StrAppend(&result, ", backend_config=\"", CEscape(backend_config_), "\"");
  }
  return result;
}

string HloInstruction::OperandsToString(const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  return OperandsToStringWithCanonicalNameMap(options, &new_map);
}

string HloInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string operands;
  if (opcode() == HloOpcode::kConstant) {
    // For constants, show the actual value in place of an empty operand list.
    //
    // In HloInstruction, sometimes a constant literal is not constructed due
    // to its size. Skip the printing in this case.
    if (HasLiteral() && ((!ShapeUtil::IsTuple(shape()) &&
                          ShapeUtil::ElementsIn(shape()) <= 10) ||
                         options.print_large_constants())) {
      // Literal::ToString emits multidimensional arrays over multiple
      // lines. Compact this into one line by stripping out white space.
      string tmp = literal().ToString();
      std::replace(tmp.begin(), tmp.end(), '\n', ' ');
      std::vector<string> v = tensorflow::str_util::Split(tmp, ' ');
      bool first = true;
      // Concatenate elements in "v" with spaces separating them, but ignoring
      // empty entries.
      for (const auto& s : v) {
        if (s.empty()) {
          continue;
        }
        StrAppend(&operands, (first ? "" : " "), s);
        first = false;
      }
    } else {
      // Do not show large constants or tuples.
      operands = "{...}";
    }
  } else if (opcode() == HloOpcode::kParameter) {
    StrAppend(&operands, parameter_number_);
  } else {
    tensorflow::gtl::ArraySlice<HloInstruction*> slice(operands_);
    const int64 kMaxOperandsToShowIfCompact = 4;
    if (options.compact_operands() &&
        slice.size() > kMaxOperandsToShowIfCompact) {
      slice.remove_suffix(slice.size() - kMaxOperandsToShowIfCompact);
    }
    operands = Join(slice, ", ", [&](string* out, HloInstruction* operand) {
      std::vector<string> str;
      if (options.print_operand_shape()) {
        str.push_back(ShapeUtil::HumanStringWithLayout(operand->shape()));
      }

      // In a top-level HloInstruction::ToString() call, the operand name is not
      // part of the canonical string.
      if (options.canonicalize_instruction_names() &&
          options.is_in_nested_computation()) {
        str.push_back(PrintName(
            canonical_name_map->LookupOrInsert(operand->name()), options));
      } else if (!options.compact_operands()) {
        str.push_back(PrintName(operand->name(), options));
      }
      StrAppend(out, Join(str, " "));
    });
    const int64 remaining = operands_.size() - slice.size();
    if (slice.size() != operands_.size()) {
      StrAppend(&operands, ", ...(+", remaining, ")");
    }
  }
  return operands;
}

std::vector<string> HloInstruction::ExtraAttributesToString(
    const HloPrintOptions& options) const {
  std::vector<string> extra;
  if (opcode() == HloOpcode::kFusion) {
    extra.push_back(StrCat("kind=", xla::ToString(fusion_kind())));
  }
  if (CanHaveDimensionsField()) {
    extra.push_back(StrCat("dimensions={", Join(dimensions(), ","), "}"));
  }
  if (window_ != nullptr && window_->dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(*window_), "}"));
  }
  if (padding_config_ != nullptr) {
    extra.push_back(
        StrCat("padding=", xla::PaddingConfigToString(*padding_config_)));
  }
  if (opcode() == HloOpcode::kSlice) {
    std::vector<string> bounds;
    bounds.reserve(slice_starts_.size());
    const bool omit_stride =
        std::all_of(slice_strides_.begin(), slice_strides_.end(),
                    [](int64 stride) { return stride == 1; });
    for (int i = 0; i < slice_starts_.size(); ++i) {
      string stride_str = omit_stride ? "" : StrCat(":", slice_strides_[i]);
      bounds.push_back(StrCat("[", slice_starts_[i], ":", slice_limits_[i],
                              stride_str, "]"));
    }
    extra.push_back(StrCat("slice={", Join(bounds, ", "), "}"));
  }
  if (opcode() == HloOpcode::kDynamicSlice) {
    extra.push_back(
        StrCat("dynamic_slice_sizes={", Join(dynamic_slice_sizes(), ","), "}"));
  }
  if (opcode() == HloOpcode::kBatchNormTraining ||
      opcode() == HloOpcode::kBatchNormInference ||
      opcode() == HloOpcode::kBatchNormGrad) {
    extra.push_back(StrCat("epsilon=", epsilon()));
    extra.push_back(StrCat("feature_index=", feature_index()));
  }

  if (convolution_dimension_numbers_ != nullptr) {
    extra.push_back(StrCat(
        "dim_labels=",
        ConvolutionDimensionNumbersToString(*convolution_dimension_numbers_)));
  }
  if (dot_dimension_numbers_ != nullptr) {
    extra.push_back(DotDimensionNumbersToString());
  }
  if (gather_dimension_numbers_ != nullptr) {
    extra.push_back(GatherDimensionNumbersToString());
    extra.push_back(
        StrCat("window_bounds={", Join(gather_window_bounds(), ","), "}"));
  }
  if (opcode() == HloOpcode::kFft) {
    extra.push_back(StrCat("fft_type=", FftType_Name(fft_type())));
    extra.push_back(StrCat("fft_length={", Join(fft_length(), ","), "}"));
  }

  if (options.print_subcomputation_mode() ==
      HloPrintOptions::PrintSubcomputationMode::kNameOnly) {
    if (opcode() == HloOpcode::kWhile) {
      extra.push_back(
          StrCat("condition=", PrintName(while_condition()->name(), options)));
      extra.push_back(
          StrCat("body=", PrintName(while_body()->name(), options)));
    } else if (opcode() == HloOpcode::kSelectAndScatter) {
      extra.push_back(StrCat("select=", PrintName(select()->name(), options)));
      extra.push_back(
          StrCat("scatter=", PrintName(scatter()->name(), options)));
    } else if (opcode() == HloOpcode::kConditional) {
      extra.push_back(StrCat("true_computation=",
                             PrintName(true_computation()->name(), options)));
      extra.push_back(StrCat("false_computation=",
                             PrintName(false_computation()->name(), options)));
    } else if (opcode() == HloOpcode::kCall || opcode() == HloOpcode::kMap ||
               opcode() == HloOpcode::kReduceWindow ||
               opcode() == HloOpcode::kReduce) {
      extra.push_back(
          StrCat("to_apply=", PrintName(to_apply()->name(), options)));
    } else if (!called_computations().empty()) {
      extra.push_back(StrCat(
          "calls=", Join(called_computations(), ", ",
                         [&](string* out, const HloComputation* computation) {
                           StrAppend(out,
                                     PrintName(computation->name(), options));
                         })));
    }
  } else if (options.print_subcomputation_mode() ==
             HloPrintOptions::PrintSubcomputationMode::kFullBodies) {
    HloPrintOptions new_options = options;
    new_options.set_is_in_nested_computation(true);
    switch (opcode()) {
      case HloOpcode::kWhile:
        extra.push_back(
            StrCat("condition=\n", while_condition()->ToString(new_options)));
        extra.push_back(StrCat("body=\n", while_body()->ToString(new_options)));
        break;
      case HloOpcode::kSelectAndScatter:
        extra.push_back(StrCat("select=\n", select()->ToString(new_options)));
        extra.push_back(StrCat("scatter=\n", scatter()->ToString(new_options)));
        break;
      case HloOpcode::kConditional:
        extra.push_back(StrCat("true_computation=\n",
                               true_computation()->ToString(new_options)));
        extra.push_back(StrCat("false_computation=\n",
                               false_computation()->ToString(new_options)));
        break;
      case HloOpcode::kCall:
      case HloOpcode::kMap:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kReduce:
        extra.push_back(
            StrCat("to_apply=\n", to_apply()->ToString(new_options)));
        break;
      default:
        if (!called_computations().empty()) {
          extra.push_back(
              StrCat("calls=\n",
                     Join(called_computations(), ", ",
                          [&](string* out, const HloComputation* computation) {
                            StrAppend(out, computation->ToString(new_options));
                          })));
        }
        break;
    }
  }
  if (opcode() == HloOpcode::kSend || opcode() == HloOpcode::kRecv ||
      opcode() == HloOpcode::kSendDone || opcode() == HloOpcode::kRecvDone) {
    extra.push_back(StrCat("channel_id=", channel_id_));
  }

  if (opcode() == HloOpcode::kGetTupleElement) {
    extra.push_back(StrCat("index=", tuple_index()));
  }
  if (has_sharding()) {
    extra.push_back(StrCat("sharding=", sharding().ToString()));
  }
  if (!control_predecessors_.empty()) {
    extra.push_back(StrCat("control-predecessors={",
                           Join(control_predecessors_, ", ",
                                [&](string* out, HloInstruction* pre) {
                                  StrAppend(out,
                                            PrintName(pre->name(), options));
                                }),
                           "}"));
  }
  if (opcode() == HloOpcode::kInfeed && !infeed_config_.empty()) {
    extra.push_back(StrCat("infeed_config=\"", CEscape(infeed_config_), "\""));
  }
  if (opcode() == HloOpcode::kOutfeed && !outfeed_config_.empty()) {
    extra.push_back(
        StrCat("outfeed_config=\"", CEscape(outfeed_config_), "\""));
  }
  if (opcode() == HloOpcode::kRng) {
    extra.push_back(
        StrCat("distribution=", RandomDistributionToString(distribution_)));
  }
  if (opcode() == HloOpcode::kReducePrecision) {
    extra.push_back(StrCat("exponent_bits=", exponent_bits_));
    extra.push_back(StrCat("mantissa_bits=", mantissa_bits_));
  }
  if (operand_side_metadata_ != nullptr) {
    extra.push_back(
        StrCat("operand_side=", operand_side_metadata_->ToString()));
  }
  if (user_side_metadata_ != nullptr) {
    extra.push_back(StrCat("user_side=", user_side_metadata_->ToString()));
  }
  // By contract, we print the custom call target even if
  // options.print_subcomputation_mode() == kOff, because the call target is not
  // an HloComputation.
  if (opcode() == HloOpcode::kCustomCall) {
    extra.push_back(
        StrCat("custom_call_target=\"", CEscape(custom_call_target_), "\""));
  }

  return extra;
}

string HloInstruction::ToShortString() const {
  return StrCat("%", name(), " = ", HloOpcodeString(opcode()), "(",
                Join(operands_, ", ",
                     [](string* out, HloInstruction* operand) {
                       StrAppend(out, "%", operand->name());
                     }),
                ")");
}

HloInstructionProto HloInstruction::ToProto() const {
  HloInstructionProto proto;
  CHECK(unique_id_ != -1)
      << "This instruction does not have a valid id. Please make sure the "
         "instruction is inside a module before dumping it.";
  proto.set_id(unique_id_);
  proto.set_name(name_);
  proto.set_opcode(HloOpcodeString(opcode_));
  *proto.mutable_shape() = shape_;
  for (const HloInstruction* operand : operands_) {
    proto.add_operand_ids(operand->unique_id());
  }
  for (const HloInstruction* control : control_predecessors_) {
    proto.add_control_predecessor_ids(control->unique_id());
  }

  *proto.mutable_metadata() = metadata_;
  proto.set_backend_config(backend_config_);
  if (literal_ != nullptr) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  proto.set_parameter_number(parameter_number_);
  if (opcode() == HloOpcode::kFusion) {
    proto.set_fusion_kind(xla::ToString(fusion_kind()));
    proto.add_called_computation_ids(
        fused_instructions_computation()->unique_id());
  } else {
    for (const HloComputation* computation : called_computations_) {
      proto.add_called_computation_ids(computation->unique_id());
    }
  }

  proto.set_tuple_index(tuple_index_);
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  if (window_ != nullptr) {
    *proto.mutable_window() = *window_;
  }
  if (convolution_dimension_numbers_ != nullptr) {
    *proto.mutable_convolution_dimension_numbers() =
        *convolution_dimension_numbers_;
  }
  if (dot_dimension_numbers_ != nullptr) {
    *proto.mutable_dot_dimension_numbers() = *dot_dimension_numbers_;
  }
  if (gather_dimension_numbers_ != nullptr) {
    *proto.mutable_gather_dimension_numbers() = *gather_dimension_numbers_;
  }
  if (opcode() == HloOpcode::kGather) {
    for (int64 bound : gather_window_bounds()) {
      proto.add_gather_window_bounds(bound);
    }
  }
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto* slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  proto.set_exponent_bits(exponent_bits_);
  proto.set_mantissa_bits(mantissa_bits_);
  for (int64 slice_size : dynamic_slice_sizes_) {
    proto.add_dynamic_slice_sizes(slice_size);
  }
  if (padding_config_ != nullptr) {
    *proto.mutable_padding_config() = *padding_config_;
  }
  proto.set_outfeed_config(outfeed_config_);
  if (opcode() == HloOpcode::kRng) {
    proto.set_distribution(distribution_);
  }
  proto.set_epsilon(epsilon_);
  proto.set_feature_index(feature_index_);
  proto.set_channel_id(channel_id_);
  proto.set_infeed_config(infeed_config_);
  proto.set_custom_call_target(custom_call_target_);
  *proto.mutable_outfeed_shape() = outfeed_shape_;
  proto.set_fft_type(fft_type_);
  for (int64 fft_len : fft_length_) {
    proto.add_fft_length(fft_len);
  }

  if (has_sharding()) {
    *proto.mutable_sharding() = sharding().ToProto();
  }

  proto.set_channel_name(channel_name_);
  proto.set_cost_estimate_ns(cost_estimate_ns_);

  return proto;
}

string HloInstruction::ToCategory() const {
  if (opcode() == HloOpcode::kTranspose || opcode() == HloOpcode::kCopy ||
      opcode() == HloOpcode::kReshape) {
    return "data formatting";
  }

  if (opcode() == HloOpcode::kConvolution) {
    string category = "convolution";
    if (window_util::HasBaseDilation(window())) {
      category += " base-dilated";
    }
    if (window_util::HasWindowDilation(window())) {
      category += " window-dilated";
    }
    return category;
  }

  // Give transpose-dot and backwards-conv fusions the categories "dot" and
  // "convolution" so they match the categories of proper kDot and kConvolution
  // ops.  These fusion categories are really just a way of expressing a
  // particular kind of dot or conv, so they should have the same category as a
  // vanilla dot/conv.
  if (opcode() == HloOpcode::kFusion) {
    switch (fusion_kind()) {
      case FusionKind::kLoop:
        return "loop fusion";
      case FusionKind::kInput:
        return "input fusion";
      case FusionKind::kOutput:
        return "output fusion";
      case FusionKind::kCustom:
        return "custom fusion";
    }
  }

  if (IsElementwise()) {
    return "non-fusion elementwise";
  }

  return HloOpcodeString(opcode());
}

HloInstruction* HloInstruction::tracing() const { return trace_instruction_; }

void HloInstruction::set_tracing(HloInstruction* trace_instruction) {
  trace_instruction_ = trace_instruction;
}

string HloInstruction::TracingTag() const {
  CHECK_EQ(HloOpcode::kTrace, opcode());
  CHECK(literal_ != nullptr);
  return literal_->GetR1U8AsString();
}

bool HloInstruction::IsFused() const { return parent_->IsFusionComputation(); }

bool HloInstruction::IsFusable() const {
  // Instructions which are traced should not be fused.
  if (tracing()) {
    return false;
  }
  // Some kinds of instructions don't make sense to fuse.
  switch (opcode_) {
    case HloOpcode::kDomain:
    case HloOpcode::kParameter:
      return false;
    // Side effecting instrutions cannot be fused.
    default:
      return !HasSideEffect();
  }
}

HloComputation* HloInstruction::fused_instructions_computation() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK(!called_computations_.empty());
  auto* fused_instructions_computation = called_computations_.front();
  CHECK(fused_instructions_computation->IsFusionComputation())
      << "Computation " << fused_instructions_computation->name()
      << " is not a fusion kind";
  return fused_instructions_computation;
}

HloInstruction* HloInstruction::fused_expression_root() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_instructions_computation()->root_instruction();
}

HloInstruction* HloInstruction::fused_parameter(int64 parameter_number) const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_instructions_computation()->parameter_instruction(
      parameter_number);
}

const std::vector<HloInstruction*>& HloInstruction::fused_parameters() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_instructions_computation()->parameter_instructions();
}

const tensorflow::gtl::iterator_range<UnwrappingIterator<
    std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
HloInstruction::fused_instructions() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  const HloComputation* subcomp = fused_instructions_computation();
  return subcomp->instructions();
}

const tensorflow::gtl::iterator_range<
    UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
HloInstruction::fused_instructions() {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_instructions_computation()->instructions();
}

int64 HloInstruction::fused_instruction_count() const {
  return fused_instructions_computation()->instruction_count();
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : unique_id_(-1),
      opcode_(opcode),
      shape_(shape),
      name_(HloOpcodeString(opcode)) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

template <typename HloInstructionPtr>
Status HloInstruction::Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor) {
  switch (opcode_) {
    case HloOpcode::kAbs:
      return visitor->HandleAbs(this);
    case HloOpcode::kAtan2:
      return visitor->HandleAtan2(this);
    case HloOpcode::kRoundNearestAfz:
      return visitor->HandleRound(this);
    case HloOpcode::kBatchNormTraining:
      return visitor->HandleBatchNormTraining(this);
    case HloOpcode::kBatchNormInference:
      return visitor->HandleBatchNormInference(this);
    case HloOpcode::kBatchNormGrad:
      return visitor->HandleBatchNormGrad(this);
    case HloOpcode::kSign:
      return visitor->HandleSign(this);
    case HloOpcode::kConstant:
      return visitor->HandleConstant(this);
    case HloOpcode::kGetTupleElement:
      return visitor->HandleGetTupleElement(this);
    case HloOpcode::kParameter:
      return visitor->HandleParameter(this);
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
      return visitor->HandleCompare(this);
    case HloOpcode::kComplex:
      return visitor->HandleComplex(this);
    case HloOpcode::kAdd:
      return visitor->HandleAdd(this);
    case HloOpcode::kDivide:
      return visitor->HandleDivide(this);
    case HloOpcode::kSubtract:
      return visitor->HandleSubtract(this);
    case HloOpcode::kMaximum:
      return visitor->HandleMaximum(this);
    case HloOpcode::kMinimum:
      return visitor->HandleMinimum(this);
    case HloOpcode::kAnd:
      return visitor->HandleAnd(this);
    case HloOpcode::kOr:
      return visitor->HandleOr(this);
    case HloOpcode::kShiftLeft:
      return visitor->HandleShiftLeft(this);
    case HloOpcode::kShiftRightArithmetic:
      return visitor->HandleShiftRightArithmetic(this);
    case HloOpcode::kShiftRightLogical:
      return visitor->HandleShiftRightLogical(this);
    case HloOpcode::kConcatenate:
      return visitor->HandleConcatenate(this);
    case HloOpcode::kConvert:
      return visitor->HandleConvert(this);
    case HloOpcode::kBitcastConvert:
      return visitor->HandleBitcastConvert(this);
    case HloOpcode::kCopy:
      return visitor->HandleCopy(this);
    case HloOpcode::kMultiply:
      return visitor->HandleMultiply(this);
    case HloOpcode::kDot:
      return visitor->HandleDot(this);
    case HloOpcode::kPower:
      return visitor->HandlePower(this);
    case HloOpcode::kRemainder:
      return visitor->HandleRemainder(this);
    case HloOpcode::kSelect:
      return visitor->HandleSelect(this);
    case HloOpcode::kConvolution:
      return visitor->HandleConvolution(this);
    case HloOpcode::kFft:
      return visitor->HandleFft(this);
    case HloOpcode::kCrossReplicaSum:
      return visitor->HandleCrossReplicaSum(this);
    case HloOpcode::kTuple:
      return visitor->HandleTuple(this);
    case HloOpcode::kMap:
      return visitor->HandleMap(this);
    case HloOpcode::kClamp:
      return visitor->HandleClamp(this);
    case HloOpcode::kReduce:
      return visitor->HandleReduce(this);
    case HloOpcode::kReduceWindow:
      return visitor->HandleReduceWindow(this);
    case HloOpcode::kSelectAndScatter:
      return visitor->HandleSelectAndScatter(this);
    case HloOpcode::kNegate:
      return visitor->HandleNegate(this);
    case HloOpcode::kExp:
      return visitor->HandleExp(this);
    case HloOpcode::kExpm1:
      return visitor->HandleExpm1(this);
    case HloOpcode::kFloor:
      return visitor->HandleFloor(this);
    case HloOpcode::kCeil:
      return visitor->HandleCeil(this);
    case HloOpcode::kClz:
      return visitor->HandleClz(this);
    case HloOpcode::kLog:
      return visitor->HandleLog(this);
    case HloOpcode::kLog1p:
      return visitor->HandleLog1p(this);
    case HloOpcode::kTanh:
      return visitor->HandleTanh(this);
    case HloOpcode::kCos:
      return visitor->HandleCos(this);
    case HloOpcode::kSin:
      return visitor->HandleSin(this);
    case HloOpcode::kReal:
      return visitor->HandleReal(this);
    case HloOpcode::kImag:
      return visitor->HandleImag(this);
    case HloOpcode::kIsFinite:
      return visitor->HandleIsFinite(this);
    case HloOpcode::kNot:
      return visitor->HandleNot(this);
    case HloOpcode::kBitcast:
      return visitor->HandleBitcast(this);
    case HloOpcode::kBroadcast:
      return visitor->HandleBroadcast(this);
    case HloOpcode::kPad:
      return visitor->HandlePad(this);
    case HloOpcode::kReshape:
      return visitor->HandleReshape(this);
    case HloOpcode::kTranspose:
      return visitor->HandleTranspose(this);
    case HloOpcode::kReverse:
      return visitor->HandleReverse(this);
    case HloOpcode::kReducePrecision:
      return visitor->HandleReducePrecision(this);
    case HloOpcode::kSlice:
      return visitor->HandleSlice(this);
    case HloOpcode::kDynamicSlice:
      return visitor->HandleDynamicSlice(this);
    case HloOpcode::kDynamicUpdateSlice:
      return visitor->HandleDynamicUpdateSlice(this);
    case HloOpcode::kSort:
      return visitor->HandleSort(this);
    case HloOpcode::kInfeed:
      return visitor->HandleInfeed(this);
    case HloOpcode::kOutfeed:
      return visitor->HandleOutfeed(this);
    case HloOpcode::kHostCompute:
      return visitor->HandleHostCompute(this);
    case HloOpcode::kRng:
      return visitor->HandleRng(this);
    case HloOpcode::kWhile:
      return visitor->HandleWhile(this);
    case HloOpcode::kFusion:
      return visitor->HandleFusion(this);
    case HloOpcode::kCall:
      return visitor->HandleCall(this);
    case HloOpcode::kConditional:
      return visitor->HandleConditional(this);
    case HloOpcode::kCustomCall:
      return visitor->HandleCustomCall(this);
    case HloOpcode::kRecv:
      return visitor->HandleRecv(this);
    case HloOpcode::kRecvDone:
      return visitor->HandleRecvDone(this);
    case HloOpcode::kSend:
      return visitor->HandleSend(this);
    case HloOpcode::kSendDone:
      return visitor->HandleSendDone(this);
    case HloOpcode::kGather:
      return visitor->HandleGather(this);
    case HloOpcode::kDomain:
      return visitor->HandleDomain(this);

    // These opcodes are not handled here.
    case HloOpcode::kTrace:
      break;
  }
  return InternalError(
      "Unhandled HloOpcode for DfsHloVisitor: %s. This should not happen - "
      "please file a bug for XLA.",
      HloOpcodeString(opcode_).c_str());
}

// Explicit instantiations.
template Status HloInstruction::Visit(DfsHloVisitor* visitor);
template Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

using DFSStack =
    tensorflow::gtl::InlinedVector<std::pair<int, HloInstruction*>, 16>;

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

using InternalCompareFunction =
    std::function<bool(std::pair<int, const HloInstruction*>,
                       std::pair<int, const HloInstruction*>)>;
template <typename Visitor>
static Status PostOrderDFS(HloInstruction* root, Visitor* visitor,
                           const InternalCompareFunction* operand_order,
                           bool ignore_control_predecessors) {
  visitor->ReserveVisitStates(root->GetModule()->NumUniqueInstructionIds());

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0) << current_id << ": " << current_node
                            << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "Not visiting HLO %" << current_node->name()
              << " as it was already visited.";
      continue;
    }

    if (visit_state == Visitor::kVisiting) {
      dfs_stack.pop_back();

      TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
      VLOG(2) << "Visiting HLO %" << current_node->name();
      TF_RETURN_IF_ERROR(current_node->Visit(visitor));
      visitor->SetVisitState(current_id, Visitor::kVisited);
      TF_RETURN_IF_ERROR(visitor->Postprocess(current_node));
      continue;
    }

    visitor->SetVisitState(current_id, Visitor::kVisiting);

    const size_t old_dfs_stack_size = dfs_stack.size();
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        return FailedPrecondition(
            "A cycle is detected while visiting instruction %s",
            current_node->ToString().c_str());
      }
    }

    if (!ignore_control_predecessors) {
      for (HloInstruction* child : current_node->control_predecessors()) {
        if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
          return FailedPrecondition(
              "A cycle is detected while visiting instruction %s",
              current_node->ToString().c_str());
        }
      }
    }

    if (operand_order != nullptr) {
      std::sort(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end(),
                *operand_order);
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

template <typename HloInstructionPtr>
Status HloInstruction::Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                              bool call_finish_visit,
                              bool ignore_control_predecessors) {
  VLOG(3) << "HloInstruction::Accept(%" << name() << ")";
  TF_RETURN_IF_ERROR(
      PostOrderDFS(this, visitor, nullptr, ignore_control_predecessors));
  if (call_finish_visit) {
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
  }
  return Status::OK();
}

// Explicit instantiations.
template Status HloInstruction::Accept(DfsHloVisitor*, bool, bool);
template Status HloInstruction::Accept(ConstDfsHloVisitor*, bool, bool);

Status HloInstruction::AcceptWithOperandOrder(
    DfsHloVisitor* visitor, const CompareFunction& operand_order,
    bool call_finish_visit) {
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder(%" << name() << ")";
  InternalCompareFunction func = [&operand_order](
                                     std::pair<int, const HloInstruction*> a,
                                     std::pair<int, const HloInstruction*> b) {
    // Call the client's comparison function on the actual HloInstruction*
    // objects (ignoring the internal ids we also have in our stack entries)
    return operand_order(a.second, b.second);
  };
  TF_RETURN_IF_ERROR(PostOrderDFS(this, visitor, &func,
                                  /*ignore_control_predecessors=*/false));
  if (call_finish_visit) {
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder BEFORE FINISH VISIT";
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder AFTER FINISH VISIT";
  }
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder EXIT";
  return Status::OK();
}

namespace {

// Returns true if the given order is a topological sort of the instructions
// it contains.
bool OrderIsTopologicalSort(const std::vector<const HloInstruction*>& order) {
  // Create a map from instruction to its position in 'order'.
  std::unordered_map<const HloInstruction*, int> order_position;
  for (int i = 0; i < order.size(); i++) {
    if (!order_position.insert({order[i], i}).second) {
      // Instruction order[i] is duplicated in the order.
      return false;
    }
  }
  // Verify that the operand of each instruction in the order is also in the
  // order *and* the operand's position is earlier (defs are before uses for
  // all ops).
  for (auto* instruction : order) {
    for (auto* operand : instruction->operands()) {
      if (!ContainsKey(order_position, operand) ||
          order_position.at(operand) >= order_position.at(instruction)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace

Status HloInstruction::Accept(
    const std::function<Status(HloInstruction*)>& visitor_func) {
  FunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

Status HloInstruction::Accept(
    const std::function<Status(const HloInstruction*)>& visitor_func) const {
  ConstFunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

Status HloInstruction::AcceptOrdered(
    DfsHloVisitor* visitor, const std::vector<const HloInstruction*>& order) {
  VLOG(2) << "HloInstruction::AcceptOrdered(%" << name() << ")";
  TF_RET_CHECK(OrderIsTopologicalSort(order));

  // Compute the predecessors of this instruction.
  std::unordered_set<const HloInstruction*> predecessors;
  TF_RETURN_IF_ERROR(this->Accept([&predecessors](HloInstruction* instruction) {
    predecessors.insert(instruction);
    return Status::OK();
  }));

  for (auto* const_instruction : order) {
    if (!ContainsKey(predecessors, const_instruction)) {
      // Instruction is not a predecessors of 'this'.
      continue;
    }

    // The visitor can mark instructions as visited to skip particular
    // instructions.
    if (visitor->DidVisit(*const_instruction)) {
      VLOG(3) << "Not visiting HLO %" << const_instruction->name()
              << " as it was already visited.";
      continue;
    }

    // TODO(b/78350259): Eliminate const laundering.
    HloInstruction* instruction =
        const_cast<HloInstruction*>(const_instruction);

    TF_RETURN_IF_ERROR(visitor->Preprocess(instruction));
    VLOG(2) << "Visiting HLO %" << instruction->name();
    TF_RETURN_IF_ERROR(instruction->Visit(visitor));
    visitor->SetVisited(*instruction);
    TF_RETURN_IF_ERROR(visitor->Postprocess(instruction));
  }

  return visitor->FinishVisit(this);
}

const Shape& HloInstruction::outfeed_shape() const {
  DCHECK_EQ(opcode_, HloOpcode::kOutfeed);
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
  return outfeed_shape_;
}

const Shape& HloInstruction::shape() const {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
  return shape_;
}

std::vector<int64> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  std::vector<int64> result;
  for (int64 i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwiseBinary() const {
  return IsElementwise() && operand_count() == 2;
}

bool HloInstruction::IsElementwise() const {
  switch (opcode_) {
    // Nullary elementwise operations.
    case HloOpcode::kConstant:
      return true;

    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTanh:
      CHECK_EQ(1, operand_count());
      return true;

    // Binary elementwise operations, the same as in IsElementwiseBinary().
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      CHECK_EQ(2, operand_count());
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
      return !ShapeUtil::IsTuple(shape_);
    case HloOpcode::kClamp:
      return true;

    // Other operations.
    case HloOpcode::kRng:
    case HloOpcode::kMap:
      return true;
    case HloOpcode::kFusion:
      if (fusion_kind() != FusionKind::kLoop) {
        return false;
      }
      for (auto* fused : fused_instructions()) {
        if (fused->opcode() != HloOpcode::kParameter &&
            !fused->IsElementwise()) {
          return false;
        }
      }
      return true;

    default:
      return false;
  }
}

bool HloInstruction::ImplicitlyBroadcastsOperand(int64 operand_idx) const {
  CHECK(IsElementwise());
  return !ShapeUtil::SameDimensions(shape(), operand(operand_idx)->shape());
}

namespace {
bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
  std::vector<int64> operand_indices = instruction->OperandIndices(operand);
  return std::all_of(
      operand_indices.begin(), operand_indices.end(),
      [instruction](int64 operand_index) {
        return instruction->IsElementwiseOnOperand(operand_index);
      });
}
}  // namespace

bool HloInstruction::IsElementwiseOnOperand(int64 operand_idx) const {
  // For all instructions other than kFusion, being elementwise on one of the
  // operands is equivalent to being elementwise on all the operands.
  if (opcode() != HloOpcode::kFusion) {
    return IsElementwise();
  }

  CHECK_EQ(HloOpcode::kFusion, opcode());
  if (fusion_kind() != FusionKind::kLoop) {
    return false;
  }

  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction*> worklist;
  std::unordered_set<const HloInstruction*> visited;
  worklist.push_back(fused_parameter(operand_idx));
  visited.insert(fused_parameter(operand_idx));
  while (!worklist.empty()) {
    HloInstruction* operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction* user : operand->users()) {
      CHECK_GE(user->unique_id(), 0);
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

// A helper class for memoized, recursive computation of HloOpcode::kFusion
// in HloInstruction::OperandElementUse below.
class HloInstruction::FusionReusesParamElements {
 public:
  using UseKind = HloInstruction::UseKind;

  // We could rather iterate backwards through fused_instructions_ here, as it
  // is in reverse postorder, and compute whether each fused instruction reuses
  // the value of this parameter, which would save stack space but not allow us
  // to finish early if we find a reuse.
  static UseKind Compute(int64 i, const HloInstruction& hlo) {
    tensorflow::gtl::FlatMap<const HloInstruction*, UseKind> memoization_cache;
    return ComputeInternal(i, hlo, &memoization_cache);
  }

 private:
  static UseKind ComputeInternal(
      int64 i, const HloInstruction& hlo,
      tensorflow::gtl::FlatMap<const HloInstruction*, UseKind>* cache) {
    if (hlo.opcode_ == HloOpcode::kParameter && hlo.parameter_number_ == i) {
      return UseKind::kUse;
    }

    auto p = cache->emplace(&hlo, UseKind{});
    auto value_it = p.first;
    const bool key_is_new = p.second;

    if (key_is_new) {
      for (int64 j = 0; j < hlo.operands_.size(); ++j) {
        UseKind old_val = value_it->second;

        // The next operation invalidates iterators.
        UseKind new_val =
            Plus(old_val, std::min(hlo.OperandElementUse(j),
                                   ComputeInternal(i, *hlo.operand(j), cache)));

        // Re-acquire the iterator. We could work harder to do this only if
        // absolutely necessary, but this code is not hot enough to warrant
        // that.
        value_it = cache->find(&hlo);
        value_it->second = new_val;
      }
    }
    return value_it->second;
  }

  // Fold operation for UseKinds.
  static UseKind Plus(UseKind a, UseKind b) {
    if (a == UseKind::kNoUse) {
      return b;
    } else if (b == UseKind::kNoUse) {
      return a;
    } else if (a == UseKind::kReuse || b == UseKind::kReuse) {
      return UseKind::kReuse;
    } else if (a == UseKind::kUsePermutingElements ||
               b == UseKind::kUsePermutingElements) {
      return UseKind::kReuse;
    } else {
      CHECK(a == UseKind::kUse && b == UseKind::kUse);
      return UseKind::kUse;
    }
  }
};

HloInstruction::UseKind HloInstruction::OperandElementUse(int64 i) const {
  switch (opcode_) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return UseKind::kUsePermutingElements;
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
      // Pad reuses the padding value but not the padded array elements.
      // Reduce reuses the init value but not the operand array elements.
      return i > 0 ? UseKind::kReuse : UseKind::kUsePermutingElements;
    case HloOpcode::kFusion:
      // Uses the memoizing, recursive computation defined above.
      return FusionReusesParamElements::Compute(i, *fused_expression_root());
    case HloOpcode::kDot:
      // Dot operations with inputs [A,B] * [B,1] do not re-use
      // elements on their left operand.
      // Dot operations with inputs [1,A] * [A,B] do not re-use
      // elements on their right operand.
      if (shape().dimensions_size() == 2) {
        if ((i == 0 && shape().dimensions(1) == 1) ||
            (i == 1 && shape().dimensions(0) == 1)) {
          return UseKind::kUse;
        }
      }
      return UseKind::kReuse;
    case HloOpcode::kDynamicUpdateSlice:
      // Dynamic-update-slice reuses only operand 2 (start_indices).
      if (i == 0 || i == 1) {
        return UseKind::kUse;
      }
      return UseKind::kReuse;
    default:
      return IsElementwise() && !ImplicitlyBroadcastsOperand(i)
                 ? UseKind::kUse
                 : UseKind::kReuse;
  }
}

std::tuple<bool, std::vector<int64>, std::vector<int64>>
HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions() const {
  if (HloOpcode::kReshape != opcode_) {
    return std::make_tuple(false, std::vector<int64>(), std::vector<int64>());
  }
  return ShapeUtil::InsertedOrDeleted1SizedDimensions(operand(0)->shape_,
                                                      shape_);
}

string ToString(HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kLoop:
      return "kLoop";
    case HloInstruction::FusionKind::kInput:
      return "kInput";
    case HloInstruction::FusionKind::kOutput:
      return "kOutput";
    case HloInstruction::FusionKind::kCustom:
      return "kCustom";
  }
}

StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    const string& kind_name) {
  if (kind_name == "kLoop") {
    return HloInstruction::FusionKind::kLoop;
  }
  if (kind_name == "kInput") {
    return HloInstruction::FusionKind::kInput;
  }
  if (kind_name == "kOutput") {
    return HloInstruction::FusionKind::kOutput;
  }
  if (kind_name == "kCustom") {
    return HloInstruction::FusionKind::kCustom;
  }
  return InvalidArgument("Unknown fusion kind: %s", kind_name.c_str());
}

string PaddingConfigToString(const PaddingConfig& padding) {
  bool has_interior_padding =
      std::any_of(padding.dimensions().begin(), padding.dimensions().end(),
                  [](const PaddingConfig::PaddingConfigDimension& dim) {
                    return dim.interior_padding() != 0;
                  });
  return Join(
      padding.dimensions(), "x",
      [&](string* out, const PaddingConfig::PaddingConfigDimension& dim) {
        StrAppend(
            out, dim.edge_padding_low(), "_", dim.edge_padding_high(),
            has_interior_padding ? StrCat("_", dim.interior_padding()) : "");
      });
}

string OpMetadataToString(const OpMetadata& metadata) {
  std::vector<string> result;
  if (!metadata.op_type().empty()) {
    result.push_back(StrCat("op_type=\"", CEscape(metadata.op_type()), "\""));
  }
  if (!metadata.op_name().empty()) {
    result.push_back(StrCat("op_name=\"", CEscape(metadata.op_name()), "\""));
  }
  if (!metadata.source_file().empty()) {
    result.push_back(
        StrCat("source_file=\"", CEscape(metadata.source_file()), "\""));
  }
  if (metadata.source_line() != 0) {
    result.push_back(StrCat("source_line=", metadata.source_line()));
  }
  return Join(result, " ");
}

string RandomDistributionToString(const RandomDistribution& distribution) {
  return tensorflow::str_util::Lowercase(RandomDistribution_Name(distribution));
}

string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums) {
  // lhs_dims[i] is the symbol of the logical dimension i for the lhs
  // operand. E.g. if batch has dimension number 2, then lhs_dims[2] == "b".
  std::vector<string> lhs_dims(2 + dnums.input_spatial_dimensions().size());
  lhs_dims[dnums.input_batch_dimension()] = 'b';
  lhs_dims[dnums.input_feature_dimension()] = 'f';
  for (int64 i = 0; i < dnums.input_spatial_dimensions().size(); ++i) {
    lhs_dims[dnums.input_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<string> rhs_dims(2 + dnums.kernel_spatial_dimensions().size());
  rhs_dims[dnums.kernel_input_feature_dimension()] = "i";
  rhs_dims[dnums.kernel_output_feature_dimension()] = "o";
  for (int64 i = 0; i < dnums.kernel_spatial_dimensions().size(); ++i) {
    rhs_dims[dnums.kernel_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<string> output_dims(2 + dnums.output_spatial_dimensions().size());
  output_dims[dnums.output_batch_dimension()] = 'b';
  output_dims[dnums.output_feature_dimension()] = 'f';
  for (int64 i = 0; i < dnums.output_spatial_dimensions().size(); ++i) {
    output_dims[dnums.output_spatial_dimensions(i)] = StrCat(i);
  }

  return StrCat(Join(lhs_dims, ""), "_", Join(rhs_dims, ""), "->",
                Join(output_dims, ""));
}

string HloInstruction::DotDimensionNumbersToString() const {
  std::vector<string> result;
  if (dot_dimension_numbers_ == nullptr) {
    return "";
  }
  const DotDimensionNumbers& dnums = *dot_dimension_numbers_;
  if (!dnums.lhs_batch_dimensions().empty()) {
    result.push_back(StrCat("lhs_batch_dims={",
                            Join(dnums.lhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("lhs_contracting_dims={",
                          Join(dnums.lhs_contracting_dimensions(), ","), "}"));

  if (!dnums.rhs_batch_dimensions().empty()) {
    result.push_back(StrCat("rhs_batch_dims={",
                            Join(dnums.rhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("rhs_contracting_dims={",
                          Join(dnums.rhs_contracting_dimensions(), ","), "}"));

  return Join(result, ", ");
}

StatusOr<RandomDistribution> StringToRandomDistribution(const string& name) {
  static std::unordered_map<string, RandomDistribution>* map = [] {
    static auto* map = new std::unordered_map<string, RandomDistribution>;
    for (int i = 0; i < RandomDistribution_ARRAYSIZE; i++) {
      if (RandomDistribution_IsValid(i)) {
        auto value = static_cast<RandomDistribution>(i);
        (*map)[RandomDistributionToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(tensorflow::str_util::Lowercase(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown distribution");
  }
  return found->second;
}

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind) {
  return os << ToString(kind);
}

string HloInstruction::GatherDimensionNumbersToString() const {
  CHECK_NE(gather_dimension_numbers_.get(), nullptr);
  string output_window_dims =
      StrCat("output_window_dims={",
             Join(gather_dimension_numbers_->output_window_dims(), ","), "}");
  string elided_window_dims =
      StrCat("elided_window_dims={",
             Join(gather_dimension_numbers_->elided_window_dims(), ","), "}");
  string gather_dims_to_operand_dims = StrCat(
      "gather_dims_to_operand_dims={",
      Join(gather_dimension_numbers_->gather_dims_to_operand_dims(), ","), "}");
  string index_vector_dim = StrCat(
      "index_vector_dim=", gather_dimension_numbers_->index_vector_dim());

  return Join<std::initializer_list<string>>(
      {output_window_dims, elided_window_dims, gather_dims_to_operand_dims,
       index_vector_dim},
      ", ");
}

bool HloInstruction::CouldBeBitcast() const {
  switch (opcode_) {
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kReshape:
      return std::get<0>(ReshapeMerelyInsertsOrDeletes1SizedDimensions());
    default:
      return false;
  }
}

Status HloInstruction::GetBackendConfigInternal(
    tensorflow::protobuf::Message* proto) const {
  proto->Clear();

  // Empty string does not parse as valid JSON, but it's a valid backend config,
  // corresponding to the empty proto.
  if (backend_config_.empty()) {
    return Status::OK();
  }
  return tensorflow::HumanReadableJsonToProto(backend_config_, proto);
}

Status HloInstruction::set_backend_config(
    const tensorflow::protobuf::Message& proto) {
  TF_ASSIGN_OR_RETURN(backend_config_, BackendConfigToRawString(proto));
  return Status::OK();
}

/* static */ StatusOr<string> HloInstruction::BackendConfigToRawString(
    const tensorflow::protobuf::Message& proto) {
  string ret;
  TF_RETURN_IF_ERROR(tensorflow::ProtoToHumanReadableJson(proto, &ret));
  return ret;
}

HloModule* HloInstruction::GetModule() const {
  if (parent_) {
    return parent_->parent();
  }
  return nullptr;
}

void HloInstruction::UniquifyName(NameUniquer* name_uniquer) {
  string parent_str = parent() == nullptr ? "noparent" : parent()->name();
  name_ = name_uniquer->GetUniqueName(name_);
}

void HloInstruction::set_outer_dimension_partitions(
    const std::vector<int64>& outer_dimension_partitions) {
  outer_dimension_partitions_ = outer_dimension_partitions;
}

void HloInstruction::RelayoutConstant(const Layout& new_layout,
                                      const ShapeIndex& shape_index) {
  CHECK_EQ(opcode(), HloOpcode::kConstant);
  Shape* mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(ShapeUtil::IsArray(*mutable_array_subshape));

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    literal_ = literal_->Relayout(new_layout, shape_index);
    *mutable_array_subshape->mutable_layout() = new_layout;
  }
}

}  // namespace xla
