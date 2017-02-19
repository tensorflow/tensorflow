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

#include "tensorflow/compiler/xla/service/user_computation.h"

#include <algorithm>
#include <set>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

HloOpcode UnaryOperationToHloOpcode(UnaryOperation unop) {
  switch (unop) {
    case UNOP_ABS:
      return HloOpcode::kAbs;
    case UNOP_CEIL:
      return HloOpcode::kCeil;
    case UNOP_EXP:
      return HloOpcode::kExp;
    case UNOP_FLOOR:
      return HloOpcode::kFloor;
    case UNOP_LOG:
      return HloOpcode::kLog;
    case UNOP_LOGICAL_NOT:
      return HloOpcode::kLogicalNot;
    case UNOP_NEGATE:
      return HloOpcode::kNegate;
    case UNOP_SIGN:
      return HloOpcode::kSign;
    case UNOP_SORT:
      return HloOpcode::kSort;
    case UNOP_TANH:
      return HloOpcode::kTanh;
    default:
      LOG(FATAL) << "unhandled operation " << unop;
  }
}

HloOpcode BinaryOperationToHloOpcode(BinaryOperation binop) {
  switch (binop) {
    case BINOP_DOT:
      return HloOpcode::kDot;
    case BINOP_MUL:
      return HloOpcode::kMultiply;
    case BINOP_ADD:
      return HloOpcode::kAdd;
    case BINOP_SUB:
      return HloOpcode::kSubtract;
    case BINOP_INDEX:
      return HloOpcode::kIndex;
    case BINOP_DIV:
      return HloOpcode::kDivide;
    case BINOP_EQ:
      return HloOpcode::kEq;
    case BINOP_GE:
      return HloOpcode::kGe;
    case BINOP_GT:
      return HloOpcode::kGt;
    case BINOP_LE:
      return HloOpcode::kLe;
    case BINOP_LT:
      return HloOpcode::kLt;
    case BINOP_NE:
      return HloOpcode::kNe;
    case BINOP_MAX:
      return HloOpcode::kMaximum;
    case BINOP_MIN:
      return HloOpcode::kMinimum;
    case BINOP_POW:
      return HloOpcode::kPower;
    case BINOP_REM:
      return HloOpcode::kRemainder;
    case BINOP_LOGICAL_OR:
      return HloOpcode::kLogicalOr;
    case BINOP_LOGICAL_AND:
      return HloOpcode::kLogicalAnd;
    default:
      LOG(FATAL) << "unhandled operation " << binop;
  }
}

HloOpcode TernaryOperationToHloOpcode(TernaryOperation triop) {
  switch (triop) {
    case TRIOP_CLAMP:
      return HloOpcode::kClamp;
    case TRIOP_SELECT:
      return HloOpcode::kSelect;
    case TRIOP_UPDATE:
      return HloOpcode::kUpdate;
    default:
      LOG(FATAL) << "unhandled operation " << triop;
  }
}

HloOpcode VariadicOperationToHloOpcode(VariadicOperation varop) {
  switch (varop) {
    case VAROP_TUPLE:
      return HloOpcode::kTuple;
    default:
      LOG(FATAL) << "unhandled operation " << varop;
  }
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<UserComputation>>
UserComputation::MakeWithRemapping(
    const SessionComputation& session_computation,
    const ComputationHandle& handle,
    const std::map<int64, ComputationHandle>& old_to_new) {
  auto user_computation =
      MakeUnique<UserComputation>(session_computation.name(), handle);
  {
    tensorflow::mutex_lock lock(user_computation->mutex_);
    user_computation->session_computation_ = session_computation;
    user_computation->next_handle_value_ =
        std::max_element(session_computation.requests().begin(),
                         session_computation.requests().end(),
                         [](const std::pair<int64, OperationRequest>& lhs,
                            const std::pair<int64, OperationRequest>& rhs) {
                           return lhs.first < rhs.first;
                         })
            ->first +
        1;
    TF_RETURN_IF_ERROR(user_computation->RemapEmbeddedComputations(old_to_new));
  }

  return std::move(user_computation);
}

UserComputation::UserComputation(const string& name,
                                 const ComputationHandle& handle)
    : name_(name), next_handle_value_(1) {
  *session_computation_.mutable_computation_handle() = handle;
  session_computation_.set_name(name);
}

ComputationDataHandle UserComputation::CreateComputationDataHandle() {
  ComputationDataHandle handle;
  handle.set_handle(next_handle_value_);
  // Handles are used as Version values and *must* be assigned consecutively for
  // computation versioning to work.
  next_handle_value_++;
  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddParameterInstruction(
    const ParameterRequest& parameter_request) {
  tensorflow::mutex_lock lock(mutex_);

  int64 parameter_number = parameter_request.parameter();
  if (parameters_.count(parameter_number) != 0) {
    return InvalidArgument("parameter %lld already registered",
                           parameter_number);
  }
  ComputationDataHandle handle = CreateComputationDataHandle();

  const Shape& validated_shape = parameter_request.shape();
  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(validated_shape));

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = validated_shape;
  *request.mutable_request()->mutable_parameter_request() = parameter_request;

  parameters_[parameter_number] = &request;

  return handle;
}

Status UserComputation::AddSendInstruction(const SendRequest& send_request) {
  tensorflow::mutex_lock lock(mutex_);

  *session_computation_.add_send_requests() = send_request;
  // Check if the operand of the instruction is valid.
  TF_RETURN_IF_ERROR(LookupRequest(send_request.operand()).status());
  return Status::OK();
}

StatusOr<ComputationDataHandle> UserComputation::AddRecvInstruction(
    const RecvRequest& recv_request) {
  tensorflow::mutex_lock lock(mutex_);

  const Shape& shape = recv_request.shape();
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
  ComputationDataHandle handle = CreateComputationDataHandle();
  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_recv_request() = recv_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddPadInstruction(
    const PadRequest& pad_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(pad_request.operand()));

  TF_ASSIGN_OR_RETURN(const OperationRequest* padding_value,
                      LookupRequest(pad_request.padding_value()));

  TF_ASSIGN_OR_RETURN(Shape inferred_shape, ShapeInference::InferPadShape(
                                                operand->output_shape(),
                                                padding_value->output_shape(),
                                                pad_request.padding_config()));

  ComputationDataHandle handle = CreateComputationDataHandle();
  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  *request.mutable_request()->mutable_pad_request() = pad_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddConstantInstruction(
    const ConstantRequest& constant_request) {
  const Shape& validated_shape = constant_request.literal().shape();
  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(validated_shape));

  tensorflow::mutex_lock lock(mutex_);

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = validated_shape;
  *request.mutable_request()->mutable_constant_request() = constant_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddGetTupleElementInstruction(
    const GetTupleElementRequest& get_tuple_element_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(get_tuple_element_request.operand()));
  Shape element_shape = ShapeUtil::GetTupleElementShape(
      operand->output_shape(), get_tuple_element_request.index());

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = element_shape;
  *request.mutable_request()->mutable_get_tuple_element_request() =
      get_tuple_element_request;

  return handle;
}

Status UserComputation::AddTraceInstruction(const TraceRequest& trace_request) {
  tensorflow::mutex_lock lock(mutex_);

  // Verify that the operand index is valid.
  TF_RETURN_IF_ERROR(LookupRequest(trace_request.operand()).status());

  *session_computation_.add_trace_requests() = trace_request;

  return Status::OK();
}

StatusOr<ComputationDataHandle> UserComputation::AddRngInstruction(
    const RngRequest& rng_request) {
  tensorflow::mutex_lock lock(mutex_);

  // Check the number of parameters per RNG distribution.
  switch (rng_request.distribution()) {
    case RandomDistribution::RNG_BERNOULLI:
      if (rng_request.parameter_size() != 1) {
        return InvalidArgument(
            "RNG distribution (%s) expects 1 parameters, but got %d",
            RandomDistribution_Name(rng_request.distribution()).c_str(),
            rng_request.parameter_size());
      }
      break;
    case RandomDistribution::RNG_NORMAL:
    case RandomDistribution::RNG_UNIFORM:
      if (rng_request.parameter_size() != 2) {
        return InvalidArgument(
            "RNG distribution (%s) expects 2 parameters, but got %d",
            RandomDistribution_Name(rng_request.distribution()).c_str(),
            rng_request.parameter_size());
      }
      break;
    default:
      LOG(FATAL) << "unhandled distribution " << rng_request.distribution();
  }

  // Verify that the parameter indices are valid;
  for (const ComputationDataHandle& param : rng_request.parameter()) {
    TF_RETURN_IF_ERROR(LookupRequest(param).status());
  }
  const Shape& validated_shape = rng_request.shape();
  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(validated_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = validated_shape;
  *request.mutable_request()->mutable_rng_request() = rng_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddMapInstruction(
    const MapRequest& map_request,
    const UserComputation& to_apply_computation) {
  tensorflow::mutex_lock lock(mutex_);

  std::vector<const Shape*> operand_shapes;
  for (const ComputationDataHandle& handle : map_request.operands()) {
    TF_ASSIGN_OR_RETURN(const OperationRequest* operand, LookupRequest(handle));
    operand_shapes.push_back(&operand->output_shape());
  }

  VersionedComputationHandle::Version to_apply_version =
      to_apply_computation.version();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> to_apply_program_shape,
      to_apply_computation.ComputeProgramShape(to_apply_version));
  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferMapShape(operand_shapes, *to_apply_program_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(to_apply_version);
  *request.mutable_request()->mutable_map_request() = map_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddReduceInstruction(
    const ReduceRequest& reduce_request,
    const UserComputation& to_apply_computation) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(reduce_request.operand()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* init_value,
                      LookupRequest(reduce_request.init_value()));

  VersionedComputationHandle::Version to_apply_version =
      to_apply_computation.version();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> to_apply_program_shape,
      to_apply_computation.ComputeProgramShape(to_apply_version));

  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferReduceShape(
          operand->output_shape(), init_value->output_shape(),
          AsInt64Slice(reduce_request.dimensions()), *to_apply_program_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(to_apply_version);
  *request.mutable_request()->mutable_reduce_request() = reduce_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddReduceWindowInstruction(
    const ReduceWindowRequest& reduce_window_request,
    const UserComputation& to_apply_computation) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(reduce_window_request.operand()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* init_value,
                      LookupRequest(reduce_window_request.init_value()));

  VersionedComputationHandle::Version to_apply_version =
      to_apply_computation.version();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> to_apply_program_shape,
      to_apply_computation.ComputeProgramShape(to_apply_version));

  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferReduceWindowShape(
          operand->output_shape(), init_value->output_shape(),
          reduce_window_request.window(), *to_apply_program_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(to_apply_version);
  *request.mutable_request()->mutable_reduce_window_request() =
      reduce_window_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddSelectAndScatterInstruction(
    const SelectAndScatterRequest& select_and_scatter_request,
    const UserComputation& select_computation,
    const UserComputation& scatter_computation) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(select_and_scatter_request.operand()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* source,
                      LookupRequest(select_and_scatter_request.source()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* init_value,
                      LookupRequest(select_and_scatter_request.init_value()));

  VersionedComputationHandle::Version select_version =
      select_computation.version();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const ProgramShape> select_program_shape,
                      select_computation.ComputeProgramShape(select_version));
  VersionedComputationHandle::Version scatter_version =
      scatter_computation.version();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const ProgramShape> scatter_program_shape,
                      scatter_computation.ComputeProgramShape(scatter_version));

  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferSelectAndScatterShape(
          operand->output_shape(), *select_program_shape,
          select_and_scatter_request.window(), source->output_shape(),
          init_value->output_shape(), *scatter_program_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(select_version);
  request.add_embedded_computation_versions(scatter_version);
  *request.mutable_request()->mutable_select_and_scatter_request() =
      select_and_scatter_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddReverseInstruction(
    const ReverseRequest& reverse_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(reverse_request.operand()));
  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferReverseShape(
          operand->output_shape(), AsInt64Slice(reverse_request.dimensions())));

  ComputationDataHandle handle = CreateComputationDataHandle();
  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  *request.mutable_request()->mutable_reverse_request() = reverse_request;
  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddWhileInstruction(
    const WhileRequest& while_request,
    const UserComputation& condition_computation,
    const UserComputation& body_computation) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* init,
                      LookupRequest(while_request.init()));

  VersionedComputationHandle::Version condition_version =
      condition_computation.version();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> condition_program_shape,
      condition_computation.ComputeProgramShape(condition_version));

  VersionedComputationHandle::Version body_version = body_computation.version();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const ProgramShape> body_program_shape,
                      body_computation.ComputeProgramShape(body_version));

  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferWhileShape(
          *condition_program_shape, *body_program_shape, init->output_shape()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(condition_version);
  request.add_embedded_computation_versions(body_version);
  *request.mutable_request()->mutable_while_request() = while_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddBroadcastInstruction(
    const BroadcastRequest& broadcast_request) {
  tensorflow::mutex_lock lock(mutex_);

  // Fetches and validates the operand.
  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(broadcast_request.operand()));
  TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                      ShapeInference::InferBroadcastShape(
                          operand->output_shape(),
                          AsInt64Slice(broadcast_request.broadcast_sizes())));

  ComputationDataHandle handle = CreateComputationDataHandle();
  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  *request.mutable_request()->mutable_broadcast_request() = broadcast_request;
  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddReshapeInstruction(
    const ReshapeRequest& reshape_request) {
  tensorflow::mutex_lock lock(mutex_);

  // Fetches and validates the operand.
  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(reshape_request.operand()));

  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferReshapeShape(
          operand->output_shape(), AsInt64Slice(reshape_request.dimensions()),
          AsInt64Slice(reshape_request.new_sizes())));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  *request.mutable_request()->mutable_reshape_request() = reshape_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddSliceInstruction(
    const SliceRequest& slice_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(slice_request.operand()));

  TF_ASSIGN_OR_RETURN(
      Shape new_shape,
      ShapeInference::InferSliceShape(
          operand->output_shape(), AsInt64Slice(slice_request.start_indices()),
          AsInt64Slice(slice_request.limit_indices())));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = new_shape;
  *request.mutable_request()->mutable_slice_request() = slice_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddDynamicSliceInstruction(
    const DynamicSliceRequest& dynamic_slice_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(dynamic_slice_request.operand()));

  TF_ASSIGN_OR_RETURN(const OperationRequest* start_indices,
                      LookupRequest(dynamic_slice_request.start_indices()));

  TF_ASSIGN_OR_RETURN(
      Shape new_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->output_shape(), start_indices->output_shape(),
          AsInt64Slice(dynamic_slice_request.slice_sizes())));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = new_shape;
  *request.mutable_request()->mutable_dynamic_slice_request() =
      dynamic_slice_request;

  return handle;
}

StatusOr<ComputationDataHandle>
UserComputation::AddDynamicUpdateSliceInstruction(
    const DynamicUpdateSliceRequest& dynamic_update_slice_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(dynamic_update_slice_request.operand()));

  TF_ASSIGN_OR_RETURN(const OperationRequest* update,
                      LookupRequest(dynamic_update_slice_request.update()));

  TF_ASSIGN_OR_RETURN(
      const OperationRequest* start_indices,
      LookupRequest(dynamic_update_slice_request.start_indices()));

  TF_ASSIGN_OR_RETURN(Shape new_shape,
                      ShapeInference::InferDynamicUpdateSliceShape(
                          operand->output_shape(), update->output_shape(),
                          start_indices->output_shape()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = new_shape;
  *request.mutable_request()->mutable_dynamic_update_slice_request() =
      dynamic_update_slice_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddConcatenateInstruction(
    const ConcatenateRequest& concatenate_request) {
  tensorflow::mutex_lock lock(mutex_);

  std::vector<const Shape*> operand_shapes;
  for (const ComputationDataHandle& handle : concatenate_request.operands()) {
    TF_ASSIGN_OR_RETURN(const OperationRequest* operand, LookupRequest(handle));
    operand_shapes.push_back(&operand->output_shape());
  }

  TF_ASSIGN_OR_RETURN(Shape new_shape,
                      ShapeInference::InferConcatOpShape(
                          operand_shapes, concatenate_request.dimension()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = new_shape;
  *request.mutable_request()->mutable_concatenate_request() =
      concatenate_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddConvertInstruction(
    const ConvertRequest& convert_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(convert_request.operand()));

  TF_ASSIGN_OR_RETURN(Shape new_shape, ShapeInference::InferConvertShape(
                                           operand->output_shape(),
                                           convert_request.new_element_type()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = new_shape;
  *request.mutable_request()->mutable_convert_request() = convert_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddConvolveInstruction(
    const ConvolveRequest& convolve_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* lhs,
                      LookupRequest(convolve_request.lhs()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* rhs,
                      LookupRequest(convolve_request.rhs()));
  TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferConvolveShape(
                                       lhs->output_shape(), rhs->output_shape(),
                                       convolve_request.window(),
                                       convolve_request.dimension_numbers()));

  const ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_convolve_request() = convolve_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddCrossReplicaSumInstruction(
    const CrossReplicaSumRequest& cross_replica_sum_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(cross_replica_sum_request.operand()));
  TF_ASSIGN_OR_RETURN(Shape shape, ShapeInference::InferCrossReplicaSumShape(
                                       operand->output_shape()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_cross_replica_sum_request() =
      cross_replica_sum_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddInfeedInstruction(
    const InfeedRequest& infeed_request) {
  tensorflow::mutex_lock lock(mutex_);

  const Shape& shape = infeed_request.shape();
  if (ShapeUtil::IsNestedTuple(shape)) {
    return InvalidArgument("Infeed does not support nested tuple shapes");
  }
  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("Given shape to Infeed must have a layout");
  }

  const ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_infeed_request() = infeed_request;

  return handle;
}

Status UserComputation::AddOutfeedInstruction(
    const OutfeedRequest& outfeed_request) {
  tensorflow::mutex_lock lock(mutex_);

  *session_computation_.add_outfeed_requests() = outfeed_request;
  // Verify that operand is valid.
  TF_RETURN_IF_ERROR(LookupRequest(outfeed_request.operand()).status());
  return Status::OK();
}

StatusOr<ComputationDataHandle> UserComputation::AddCallInstruction(
    const CallRequest& call_request,
    const UserComputation& to_apply_computation) {
  tensorflow::mutex_lock lock(mutex_);

  std::vector<const Shape*> operand_shapes;
  for (const ComputationDataHandle& handle : call_request.operands()) {
    TF_ASSIGN_OR_RETURN(const OperationRequest* operand, LookupRequest(handle));
    operand_shapes.push_back(&operand->output_shape());
  }

  VersionedComputationHandle::Version to_apply_version =
      to_apply_computation.version();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> to_apply_program_shape,
      to_apply_computation.ComputeProgramShape(to_apply_version));
  TF_ASSIGN_OR_RETURN(
      Shape inferred_shape,
      ShapeInference::InferCallShape(operand_shapes, *to_apply_program_shape));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = inferred_shape;
  request.add_embedded_computation_versions(to_apply_version);
  *request.mutable_request()->mutable_call_request() = call_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddCustomCallInstruction(
    const CustomCallRequest& custom_call_request) {
  tensorflow::mutex_lock lock(mutex_);

  for (const ComputationDataHandle& handle : custom_call_request.operands()) {
    TF_RETURN_IF_ERROR(LookupRequest(handle).status());
  }

  const ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = custom_call_request.shape();
  *request.mutable_request()->mutable_custom_call_request() =
      custom_call_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddUnaryInstruction(
    const UnaryOpRequest& unary_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand,
                      LookupRequest(unary_request.operand()));
  TF_ASSIGN_OR_RETURN(
      Shape shape, ShapeInference::InferUnaryOpShape(unary_request.unop(),
                                                     operand->output_shape()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_unary_op_request() = unary_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddBinaryInstruction(
    const BinaryOpRequest& binary_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* lhs,
                      LookupRequest(binary_request.lhs()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* rhs,
                      LookupRequest(binary_request.rhs()));
  TF_ASSIGN_OR_RETURN(
      Shape shape,
      ShapeInference::InferBinaryOpShape(
          binary_request.binop(), lhs->output_shape(), rhs->output_shape(),
          AsInt64Slice(binary_request.broadcast_dimensions())));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_binary_op_request() = binary_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddTernaryInstruction(
    const TernaryOpRequest& ternary_request) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* lhs,
                      LookupRequest(ternary_request.lhs()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* rhs,
                      LookupRequest(ternary_request.rhs()));
  TF_ASSIGN_OR_RETURN(const OperationRequest* ehs,
                      LookupRequest(ternary_request.ehs()));
  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeInference::InferTernaryOpShape(
                          ternary_request.triop(), lhs->output_shape(),
                          rhs->output_shape(), ehs->output_shape()));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_ternary_op_request() = ternary_request;

  return handle;
}

StatusOr<ComputationDataHandle> UserComputation::AddVariadicInstruction(
    const VariadicOpRequest& variadic_request) {
  tensorflow::mutex_lock lock(mutex_);

  std::vector<const Shape*> operand_shapes;
  for (const ComputationDataHandle& handle : variadic_request.operands()) {
    TF_ASSIGN_OR_RETURN(const OperationRequest* operand, LookupRequest(handle));
    operand_shapes.push_back(&operand->output_shape());
  }

  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeInference::InferVariadicOpShape(
                          variadic_request.varop(), operand_shapes));

  ComputationDataHandle handle = CreateComputationDataHandle();

  OperationRequest& request =
      (*session_computation_.mutable_requests())[handle.handle()];
  *request.mutable_output_handle() = handle;
  *request.mutable_output_shape() = shape;
  *request.mutable_request()->mutable_variadic_op_request() = variadic_request;

  return handle;
}

StatusOr<Shape> UserComputation::GetShape(const ComputationDataHandle& handle) {
  tensorflow::mutex_lock lock(mutex_);

  TF_ASSIGN_OR_RETURN(const OperationRequest* operand, LookupRequest(handle));
  return operand->output_shape();
}

Status UserComputation::SetReturnValue(const ComputationDataHandle& handle) {
  tensorflow::mutex_lock lock(mutex_);

  if (!(handle.handle() > 0 && handle.handle() < next_handle_value_)) {
    return InvalidArgument("Invalid handle in SetReturnValue");
  }

  handle_to_return_ = handle;

  return Status::OK();
}

VersionedComputationHandle UserComputation::GetVersionedHandle() const {
  tensorflow::mutex_lock lock(mutex_);

  VersionedComputationHandle versioned_handle;
  versioned_handle.handle = session_computation_.computation_handle();

  if (handle_to_return_.handle() > 0) {
    // A specific handle has been requested for the result of the computation.
    versioned_handle.version = handle_to_return_.handle();
  } else {
    // A version value is simply the most recently assigned
    // ComputationDataHandle value, ie the handle value of the root of the
    // computation.
    versioned_handle.version = next_handle_value_ - 1;
  }

  return versioned_handle;
}

VersionedComputationHandle UserComputation::GetVersionedHandleAtOperation(
    const ComputationDataHandle& operation) const {
  tensorflow::mutex_lock lock(mutex_);

  // The version at which an operation was added is simply the handle value of
  // the ComputationDataHandle.
  VersionedComputationHandle versioned_handle;
  versioned_handle.handle = session_computation_.computation_handle();
  versioned_handle.version = operation.handle();
  return versioned_handle;
}

VersionedComputationHandle::Version UserComputation::version() const {
  return GetVersionedHandle().version;
}

StatusOr<std::shared_ptr<const ProgramShape>>
UserComputation::ComputeProgramShape(
    VersionedComputationHandle::Version version) const {
  tensorflow::mutex_lock lock(mutex_);

  CHECK(version > 0 && version < next_handle_value_);

  if (program_shape_ == nullptr || program_shape_version_ != version) {
    // ProgramShape has not been computed yet, or is for different
    // version. Compute it now.
    TF_RETURN_IF_ERROR(CheckParametersAreContiguous(version));

    auto program_shape = MakeUnique<ProgramShape>();
    for (int64 request_num = 1; request_num <= version; ++request_num) {
      const OperationRequest& request =
          session_computation_.requests().at(request_num);
      if (request.request().op_case() == OpRequest::kParameterRequest) {
        const ParameterRequest& parameter_request =
            request.request().parameter_request();
        int64 param_no = parameter_request.parameter();
        // Parameters may be out of order so expand ProgramShape parameters
        // until
        // it is at least large enough to hold the current parameter number.
        while (program_shape->parameters_size() <= param_no) {
          program_shape->add_parameters();
          program_shape->add_parameter_names();
        }
        *program_shape->mutable_parameters(param_no) = request.output_shape();
        *program_shape->mutable_parameter_names(param_no) =
            parameter_request.name();
      }
    }

    // The root determines the output shape.
    *program_shape->mutable_result() = GetRoot(version).output_shape();
    if (ShapeUtil::IsOpaque(program_shape->result())) {
      return Unimplemented("Computation results cannot be opaque");
    }

    program_shape_ = std::move(program_shape);
    program_shape_version_ = version;
  }

  return program_shape_;
}

namespace {

// A visitor which checks whether an operation is a compile-time constant. That
// is, the operation does not depend on any parameter instructions. The visitor
// walks the computation starting at a given operation and sets is_constant to
// false iff a parameter or RNG operation is encountered.
void ConstantVisitor(const SessionComputation& session_computation,
                     const ComputationDataHandle& handle,
                     std::set<int64>* visited, bool* is_constant) {
  if (visited->count(handle.handle()) != 0 || !*is_constant) {
    return;
  }

  const OperationRequest& request =
      session_computation.requests().at(handle.handle());
  switch (request.request().op_case()) {
    case OpRequest::kRngRequest:
      *is_constant = false;
      break;

    case OpRequest::kConstantRequest:
      break;

    case OpRequest::kGetTupleElementRequest: {
      const GetTupleElementRequest& get_tuple_element_request =
          request.request().get_tuple_element_request();
      ConstantVisitor(session_computation, get_tuple_element_request.operand(),
                      visited, is_constant);
      break;
    }

    case OpRequest::kSliceRequest: {
      const SliceRequest& slice_request = request.request().slice_request();
      ConstantVisitor(session_computation, slice_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kDynamicSliceRequest: {
      const DynamicSliceRequest& dynamic_slice_request =
          request.request().dynamic_slice_request();
      ConstantVisitor(session_computation, dynamic_slice_request.operand(),
                      visited, is_constant);
      ConstantVisitor(session_computation,
                      dynamic_slice_request.start_indices(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kDynamicUpdateSliceRequest: {
      const DynamicUpdateSliceRequest& dynamic_update_slice_request =
          request.request().dynamic_update_slice_request();
      ConstantVisitor(session_computation,
                      dynamic_update_slice_request.operand(), visited,
                      is_constant);
      ConstantVisitor(session_computation,
                      dynamic_update_slice_request.update(), visited,
                      is_constant);
      ConstantVisitor(session_computation,
                      dynamic_update_slice_request.start_indices(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kConcatenateRequest: {
      const ConcatenateRequest& concatenate_request =
          request.request().concatenate_request();
      for (const ComputationDataHandle& handle :
           concatenate_request.operands()) {
        ConstantVisitor(session_computation, handle, visited, is_constant);
      }
      break;
    }

    case OpRequest::kConvolveRequest: {
      const ConvolveRequest& convolve_request =
          request.request().convolve_request();
      ConstantVisitor(session_computation, convolve_request.lhs(), visited,
                      is_constant);
      ConstantVisitor(session_computation, convolve_request.rhs(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kCrossReplicaSumRequest: {
      // TODO(b/33009255): Implmement constant folding for cross replica sum.
      *is_constant = false;
      break;
    }

    case OpRequest::kInfeedRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kOutfeedRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kCallRequest: {
      const CallRequest& call_request = request.request().call_request();
      for (const ComputationDataHandle& handle : call_request.operands()) {
        ConstantVisitor(session_computation, handle, visited, is_constant);
      }
      // TODO(b/32495713): We aren't checking the to_apply computation itself,
      // so we conservatively say that computations containing the Call op
      // cannot be constant.  We cannot set is_constant=false in other similar
      // cases since we're already relying on IsConstant to return true.
      *is_constant = false;
      break;
    }

    case OpRequest::kCustomCallRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kSendRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kRecvRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kMapRequest: {
      const MapRequest& map_request = request.request().map_request();
      for (const ComputationDataHandle& handle : map_request.operands()) {
        ConstantVisitor(session_computation, handle, visited, is_constant);
      }
      // TODO(b/32495713): We aren't checking the to_apply computation itself.
      break;
    }

    case OpRequest::kReduceRequest: {
      const ReduceRequest& reduce_request = request.request().reduce_request();
      ConstantVisitor(session_computation, reduce_request.operand(), visited,
                      is_constant);
      ConstantVisitor(session_computation, reduce_request.init_value(), visited,
                      is_constant);
      // TODO(b/32495713): We aren't checking the to_apply computation itself.
      break;
    }

    case OpRequest::kReduceWindowRequest: {
      const ReduceWindowRequest& reduce_window_request =
          request.request().reduce_window_request();
      ConstantVisitor(session_computation, reduce_window_request.operand(),
                      visited, is_constant);
      ConstantVisitor(session_computation, reduce_window_request.init_value(),
                      visited, is_constant);
      // TODO(b/32495713): We aren't checking the to_apply computation itself.
      break;
    }

    case OpRequest::kSelectAndScatterRequest: {
      const SelectAndScatterRequest& select_and_scatter_request =
          request.request().select_and_scatter_request();
      ConstantVisitor(session_computation, select_and_scatter_request.operand(),
                      visited, is_constant);
      ConstantVisitor(session_computation, select_and_scatter_request.source(),
                      visited, is_constant);
      ConstantVisitor(session_computation,
                      select_and_scatter_request.init_value(), visited,
                      is_constant);
      // TODO(b/32495713): We aren't checking the select and scatter
      // computations themselves.
      break;
    }

    case OpRequest::kBroadcastRequest: {
      const BroadcastRequest& broadcast_request =
          request.request().broadcast_request();
      ConstantVisitor(session_computation, broadcast_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kReshapeRequest: {
      const ReshapeRequest& reshape_request =
          request.request().reshape_request();
      ConstantVisitor(session_computation, reshape_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kReverseRequest: {
      const ReverseRequest& reverse_request =
          request.request().reverse_request();
      ConstantVisitor(session_computation, reverse_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kPadRequest: {
      const PadRequest& pad_request = request.request().pad_request();
      ConstantVisitor(session_computation, pad_request.operand(), visited,
                      is_constant);
      ConstantVisitor(session_computation, pad_request.padding_value(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kParameterRequest: {
      *is_constant = false;
      break;
    }

    case OpRequest::kConvertRequest: {
      const ConvertRequest& convert_request =
          request.request().convert_request();
      ConstantVisitor(session_computation, convert_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kWhileRequest: {
      const WhileRequest& while_request = request.request().while_request();
      ConstantVisitor(session_computation, while_request.init(), visited,
                      is_constant);
      // TODO(b/32495713): We aren't checking the condition and body
      // computations themselves.
      break;
    }

    case OpRequest::kTernaryOpRequest: {
      const TernaryOpRequest& ternary_op_request =
          request.request().ternary_op_request();
      ConstantVisitor(session_computation, ternary_op_request.lhs(), visited,
                      is_constant);
      ConstantVisitor(session_computation, ternary_op_request.rhs(), visited,
                      is_constant);
      ConstantVisitor(session_computation, ternary_op_request.ehs(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kVariadicOpRequest: {
      const VariadicOpRequest& variadic_op_request =
          request.request().variadic_op_request();
      for (const ComputationDataHandle& handle :
           variadic_op_request.operands()) {
        ConstantVisitor(session_computation, handle, visited, is_constant);
      }
      break;
    }

    case OpRequest::kUnaryOpRequest: {
      const UnaryOpRequest& unary_op_request =
          request.request().unary_op_request();
      ConstantVisitor(session_computation, unary_op_request.operand(), visited,
                      is_constant);
      break;
    }

    case OpRequest::kBinaryOpRequest: {
      const BinaryOpRequest& binary_op_request =
          request.request().binary_op_request();
      ConstantVisitor(session_computation, binary_op_request.lhs(), visited,
                      is_constant);
      ConstantVisitor(session_computation, binary_op_request.rhs(), visited,
                      is_constant);
      break;
    }

    case OpRequest::OP_NOT_SET:
      LOG(FATAL) << "OperationRequest doesn't contain a request";

    default:
      LOG(FATAL) << "Unexpected request type: " << request.request().op_case();
  }
  visited->insert(handle.handle());
}

}  // namespace

StatusOr<bool> UserComputation::IsConstant(
    const ComputationDataHandle& handle) {
  tensorflow::mutex_lock lock(mutex_);

  // Verify that the handle is valid.
  auto operation_status = LookupRequest(handle);
  if (!operation_status.ok()) {
    return operation_status.status();
  }

  bool is_constant = true;
  std::set<int64> visited;
  ConstantVisitor(session_computation_, handle, &visited, &is_constant);

  return is_constant;
}

const OperationRequest& UserComputation::GetRoot(
    VersionedComputationHandle::Version version) const {
  CHECK(version > 0 && version < next_handle_value_);
  return session_computation_.requests().at(version);
}

std::vector<VersionedComputationHandle>
UserComputation::GetEmbeddedComputations(
    VersionedComputationHandle::Version version) const {
  tensorflow::mutex_lock lock(mutex_);

  std::vector<VersionedComputationHandle> computations;
  for (const auto& handle_request : session_computation_.requests()) {
    int64 handle_value = handle_request.first;
    if (handle_value <= version) {
      const OperationRequest& request = handle_request.second;
      switch (request.request().op_case()) {
        case OpRequest::kCallRequest: {
          CHECK_EQ(1, request.embedded_computation_versions_size());
          const CallRequest& call_request = request.request().call_request();
          const VersionedComputationHandle versioned_handle = {
              call_request.to_apply(),
              request.embedded_computation_versions(0)};
          computations.push_back(versioned_handle);
          break;
        }

        case OpRequest::kMapRequest: {
          CHECK_EQ(1, request.embedded_computation_versions_size());
          const MapRequest& map_request = request.request().map_request();
          const VersionedComputationHandle versioned_handle = {
              map_request.to_apply(), request.embedded_computation_versions(0)};
          computations.push_back(versioned_handle);
          break;
        }

        case OpRequest::kReduceRequest: {
          CHECK_EQ(1, request.embedded_computation_versions_size());
          const ReduceRequest& reduce_request =
              request.request().reduce_request();
          const VersionedComputationHandle versioned_handle = {
              reduce_request.to_apply(),
              request.embedded_computation_versions(0)};
          computations.push_back(versioned_handle);
          break;
        }

        case OpRequest::kReduceWindowRequest: {
          CHECK_EQ(1, request.embedded_computation_versions_size());
          const ReduceWindowRequest& reduce_window_request =
              request.request().reduce_window_request();
          const VersionedComputationHandle versioned_handle = {
              reduce_window_request.to_apply(),
              request.embedded_computation_versions(0)};
          computations.push_back(versioned_handle);
          break;
        }

        case OpRequest::kSelectAndScatterRequest: {
          CHECK_EQ(2, request.embedded_computation_versions_size());
          const SelectAndScatterRequest& select_and_scatter_request =
              request.request().select_and_scatter_request();
          const VersionedComputationHandle select_versioned_handle = {
              select_and_scatter_request.select(),
              request.embedded_computation_versions(0)};
          computations.push_back(select_versioned_handle);
          const VersionedComputationHandle scatter_versioned_handle = {
              select_and_scatter_request.scatter(),
              request.embedded_computation_versions(1)};
          computations.push_back(scatter_versioned_handle);
          break;
        }

        case OpRequest::kWhileRequest: {
          CHECK_EQ(2, request.embedded_computation_versions_size());
          const WhileRequest& while_request = request.request().while_request();
          const VersionedComputationHandle condition_versioned_handle = {
              while_request.condition(),
              request.embedded_computation_versions(0)};
          computations.push_back(condition_versioned_handle);
          const VersionedComputationHandle body_versioned_handle = {
              while_request.body(), request.embedded_computation_versions(1)};
          computations.push_back(body_versioned_handle);
          break;
        }

        default:
          // No embedded computation.
          break;
      }
    }
  }
  return computations;
}

Status UserComputation::RemapEmbeddedComputations(
    const std::map<int64, ComputationHandle>& old_to_new) {
  auto update = [&old_to_new](ComputationHandle* to_update) -> Status {
    int64 old = to_update->handle();
    auto it = old_to_new.find(old);
    if (it == old_to_new.end()) {
      string mapping = tensorflow::str_util::Join(
          old_to_new, ", ",
          [](string* out, std::pair<int64, ComputationHandle> element) {
            tensorflow::strings::Appendf(out, "%lld:%lld", element.first,
                                         element.second.handle());
          });
      return NotFound(
          "could not find referenced (old) computation handle in mapping: "
          "%lld; mapping: {%s}",
          old, mapping.c_str());
    }
    VLOG(2) << "remapping " << old << " to " << it->second.handle();
    *to_update = it->second;
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(update(session_computation_.mutable_computation_handle()));
  for (auto& handle_request : *session_computation_.mutable_requests()) {
    OperationRequest& request = handle_request.second;
    switch (request.request().op_case()) {
      case OpRequest::kCallRequest: {
        TF_RET_CHECK(1 == request.embedded_computation_versions_size());
        CallRequest* call_request =
            request.mutable_request()->mutable_call_request();
        TF_RETURN_IF_ERROR(update(call_request->mutable_to_apply()));
        break;
      }
      case OpRequest::kMapRequest: {
        TF_RET_CHECK(1 == request.embedded_computation_versions_size());
        MapRequest* map_request =
            request.mutable_request()->mutable_map_request();
        TF_RETURN_IF_ERROR(update(map_request->mutable_to_apply()));
        break;
      }
      case OpRequest::kReduceRequest: {
        TF_RET_CHECK(1 == request.embedded_computation_versions_size());
        ReduceRequest* reduce_request =
            request.mutable_request()->mutable_reduce_request();
        TF_RETURN_IF_ERROR(update(reduce_request->mutable_to_apply()));
        break;
      }
      case OpRequest::kReduceWindowRequest: {
        TF_RET_CHECK(1 == request.embedded_computation_versions_size());
        ReduceWindowRequest* reduce_window_request =
            request.mutable_request()->mutable_reduce_window_request();
        TF_RETURN_IF_ERROR(update(reduce_window_request->mutable_to_apply()));
        break;
      }
      case OpRequest::kSelectAndScatterRequest: {
        TF_RET_CHECK(2 == request.embedded_computation_versions_size());
        SelectAndScatterRequest* select_and_scatter_request =
            request.mutable_request()->mutable_select_and_scatter_request();
        TF_RETURN_IF_ERROR(
            update(select_and_scatter_request->mutable_select()));
        TF_RETURN_IF_ERROR(
            update(select_and_scatter_request->mutable_scatter()));
        break;
      }
      case OpRequest::kWhileRequest: {
        TF_RET_CHECK(2 == request.embedded_computation_versions_size());
        WhileRequest* while_request =
            request.mutable_request()->mutable_while_request();
        TF_RETURN_IF_ERROR(update(while_request->mutable_condition()));
        TF_RETURN_IF_ERROR(update(while_request->mutable_body()));
        break;
      }
      default:
        // No embedded computation.
        TF_RET_CHECK(0 == request.embedded_computation_versions_size());
        break;
    }
  }
  return Status::OK();
}

SessionComputation UserComputation::CloneSessionComputation(
    VersionedComputationHandle::Version version) const {
  tensorflow::mutex_lock lock(mutex_);
  SessionComputation result = session_computation_;
  // Erase all the requests that exceed the version specified.
  // There's no lower_bound method on tensorflow::protobuf::Map so we iterate
  // all the elements.
  auto it = result.mutable_requests()->begin();
  while (it != result.mutable_requests()->end()) {
    if (it->first > version) {
      it = result.mutable_requests()->erase(it);
    } else {
      ++it;
    }
  }
  return result;
}

StatusOr<const OperationRequest*> UserComputation::LookupRequest(
    const ComputationDataHandle& handle) const {
  int64 handle_value = handle.handle();
  if (session_computation_.requests().count(handle_value) == 0) {
    return InvalidArgument("no ComputationDataHandle value %lld", handle_value);
  }
  return &session_computation_.requests().at(handle_value);
}

Status UserComputation::CheckParametersAreContiguous(
    VersionedComputationHandle::Version version) const {
  TF_RET_CHECK(version > 0 && version < next_handle_value_);

  // Determine number of parameter inputs at the given version.
  std::map<int64, const ParameterRequest*> parameter_requests;
  for (int64 request_num = 1; request_num <= version; ++request_num) {
    const OperationRequest& request =
        session_computation_.requests().at(request_num);

    if (request.request().op_case() == OpRequest::kParameterRequest) {
      const ParameterRequest& parameter_request =
          request.request().parameter_request();
      // Duplicate parameters should be checked when parameter requests are
      // added.
      TF_RET_CHECK(0 ==
                   parameter_requests.count(parameter_request.parameter()));
      parameter_requests[parameter_request.parameter()] = &parameter_request;
    }
  }

  auto program_shape = MakeUnique<ProgramShape>();
  for (int64 i = 0; i < parameter_requests.size(); ++i) {
    auto it = parameter_requests.find(i);
    if (it == parameter_requests.end()) {
      return FailedPrecondition(
          "computation %s does not have all its parameters populated "
          "sequentially, missing parameter %lld",
          name_.c_str(), i);
    }
  }

  return Status::OK();
}

namespace {

// Helper class which builds an HLO computation from a SessionComputation. To
// construct the HLO computation, the SessionComputation graph is walked in
// DFS order lowering each OperationRequest to an HLO instruction.
class ComputationLowerer {
 public:
  static std::unique_ptr<HloComputation> Lower(
      const string& computation_name,
      const SessionComputation& session_computation,
      VersionedComputationHandle::Version version,
      UserComputation::HloComputationResolver hlo_resolver,
      bool include_unused_parameters) {
    ComputationLowerer lowerer(computation_name, session_computation, version,
                               std::move(hlo_resolver));
    return lowerer.Lower(include_unused_parameters);
  }

 private:
  ComputationLowerer(const string& computation_name,
                     const SessionComputation& session_computation,
                     VersionedComputationHandle::Version version,
                     UserComputation::HloComputationResolver hlo_resolver)
      : hlo_builder_(computation_name),
        session_computation_(session_computation),
        version_(version),
        hlo_resolver_(std::move(hlo_resolver)) {}

  // Build an HLO computation from the SessionComputation at the given
  // version.
  std::unique_ptr<HloComputation> Lower(bool include_unused_parameters);

 private:
  // DFS visitor of the UserComputation operations which lowers the operations
  // to HLO instructions.
  HloInstruction* Visit(const ComputationDataHandle& handle,
                        std::map<int64, HloInstruction*>* visited);

  // Resolves a ComputationHandle and Version to a previously lowered
  // HloComputation using the hlo_resolver_ function.
  HloComputation* ResolveComputation(
      const ComputationHandle& handle,
      VersionedComputationHandle::Version version);

  HloComputation::Builder hlo_builder_;
  const SessionComputation& session_computation_;
  const VersionedComputationHandle::Version version_;
  const UserComputation::HloComputationResolver hlo_resolver_;
};

std::unique_ptr<HloComputation> ComputationLowerer::Lower(
    bool include_unused_parameters) {
  // Map from ComputationDataHandle to HLO instruction. Serves as a record of
  // which operations have been visited as well as a cache for looking up
  // ComputationDataHandles as HloInstructions.
  std::map<int64, HloInstruction*> visited;

  // A version is simply a ComputationDataHandle of the root of the computation
  // at the time the version was generated. Create a ComputationDataHandle with
  // this value and pass it to the visitor as the root of the computation to
  // lower.
  ComputationDataHandle root_handle;
  root_handle.set_handle(version_);

  HloInstruction* hlo_root = Visit(root_handle, &visited);

  // A computation may have unused parameters.
  if (include_unused_parameters) {
    for (int64 request_num = 1; request_num <= version_; ++request_num) {
      const OperationRequest& request =
          session_computation_.requests().at(request_num);
      if (request.request().op_case() == OpRequest::kParameterRequest &&
          visited.count(request.output_handle().handle()) == 0) {
        Visit(request.output_handle(), &visited);
      }
    }
  }

  // Add trace instructions.
  for (const auto& trace_request : session_computation_.trace_requests()) {
    if (trace_request.operand().handle() <= version_) {
      HloInstruction* operand = visited[trace_request.operand().handle()];
      // Trace instructions cannot be the root of a computation.
      HloInstruction* trace_instruction = hlo_builder_.AddInstruction(
          HloInstruction::CreateTrace(trace_request.tag(), operand));
      operand->set_tracing(trace_instruction);
    }
  }

  // Send instructions do not have users, so they are not reachable from the
  // root instruction. Therefore, explicitly visit all Send requests (and their
  // operand chains) and add to the builder.
  for (const auto& send_request : session_computation_.send_requests()) {
    Visit(send_request.operand(), &visited);
    HloInstruction* operand = visited[send_request.operand().handle()];
    hlo_builder_.AddInstruction(HloInstruction::CreateSend(
        operand, send_request.channel_handle().handle()));
  }

  // Outfeed instructions do not have users. Explicitly visit all Outfeed
  // requests (and their operand chains).
  for (const auto& outfeed_request : session_computation_.outfeed_requests()) {
    Visit(outfeed_request.operand(), &visited);
    HloInstruction* operand = visited[outfeed_request.operand().handle()];
    hlo_builder_.AddInstruction(HloInstruction::CreateOutfeed(
        operand, outfeed_request.outfeed_config()));
  }

  return hlo_builder_.Build(hlo_root);
}

HloComputation* ComputationLowerer::ResolveComputation(
    const ComputationHandle& handle,
    VersionedComputationHandle::Version version) {
  const VersionedComputationHandle checked_handle = {handle, version};
  return hlo_resolver_(checked_handle);
}

HloInstruction* ComputationLowerer::Visit(
    const ComputationDataHandle& handle,
    std::map<int64, HloInstruction*>* visited) {
  if (visited->count(handle.handle()) != 0) {
    return (*visited)[handle.handle()];
  }

  const OperationRequest& request =
      session_computation_.requests().at(handle.handle());
  HloInstruction* hlo_instruction;
  switch (request.request().op_case()) {
    case OpRequest::kRngRequest: {
      const RngRequest& rng_request = request.request().rng_request();
      std::vector<HloInstruction*> parameters;
      for (const ComputationDataHandle& param : rng_request.parameter()) {
        parameters.push_back(Visit(param, visited));
      }
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateRng(
          request.output_shape(), rng_request.distribution(), parameters));
      break;
    }

    case OpRequest::kConstantRequest: {
      const ConstantRequest& constant_request =
          request.request().constant_request();
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CloneToUnique(constant_request.literal())));
      break;
    }

    case OpRequest::kGetTupleElementRequest: {
      const GetTupleElementRequest& get_tuple_element_request =
          request.request().get_tuple_element_request();
      HloInstruction* operand =
          Visit(get_tuple_element_request.operand(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateGetTupleElement(
              request.output_shape(), operand,
              get_tuple_element_request.index()));
      break;
    }

    case OpRequest::kSliceRequest: {
      const SliceRequest& slice_request = request.request().slice_request();
      HloInstruction* operand = Visit(slice_request.operand(), visited);
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateSlice(
          request.output_shape(), operand,
          AsInt64Slice(slice_request.start_indices()),
          AsInt64Slice(slice_request.limit_indices())));
      break;
    }

    case OpRequest::kDynamicSliceRequest: {
      const DynamicSliceRequest& dynamic_slice_request =
          request.request().dynamic_slice_request();
      HloInstruction* operand = Visit(dynamic_slice_request.operand(), visited);
      HloInstruction* start_indices =
          Visit(dynamic_slice_request.start_indices(), visited);

      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
              request.output_shape(), operand, start_indices,
              AsInt64Slice(dynamic_slice_request.slice_sizes())));
      break;
    }

    case OpRequest::kDynamicUpdateSliceRequest: {
      const DynamicUpdateSliceRequest& dynamic_update_slice_request =
          request.request().dynamic_update_slice_request();
      HloInstruction* operand =
          Visit(dynamic_update_slice_request.operand(), visited);
      HloInstruction* update =
          Visit(dynamic_update_slice_request.update(), visited);
      HloInstruction* start_indices =
          Visit(dynamic_update_slice_request.start_indices(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              request.output_shape(), operand, update, start_indices));
      break;
    }

    case OpRequest::kConcatenateRequest: {
      const ConcatenateRequest& concatenate_request =
          request.request().concatenate_request();
      std::vector<HloInstruction*> operands;
      for (const ComputationDataHandle& handle :
           concatenate_request.operands()) {
        HloInstruction* operand = Visit(handle, visited);
        operands.push_back(operand);
      }
      hlo_instruction = hlo_builder_.AddInstruction(
          HloInstruction::CreateConcatenate(request.output_shape(), operands,
                                            concatenate_request.dimension()));
      break;
    }

    case OpRequest::kConvolveRequest: {
      const ConvolveRequest& convolve_request =
          request.request().convolve_request();
      HloInstruction* lhs = Visit(convolve_request.lhs(), visited);
      HloInstruction* rhs = Visit(convolve_request.rhs(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateConvolve(
              request.output_shape(), lhs, rhs, convolve_request.window(),
              convolve_request.dimension_numbers()));
      break;
    }

    case OpRequest::kCrossReplicaSumRequest: {
      const CrossReplicaSumRequest& cross_replica_sum_request =
          request.request().cross_replica_sum_request();
      HloInstruction* operand =
          Visit(cross_replica_sum_request.operand(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateCrossReplicaSum(
              request.output_shape(), operand));
      break;
    }

    case OpRequest::kInfeedRequest: {
      const InfeedRequest& infeed_request = request.request().infeed_request();
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateInfeed(
              request.output_shape(), infeed_request.config()));
      break;
    }

    case OpRequest::kOutfeedRequest: {
      const OutfeedRequest& outfeed_request =
          request.request().outfeed_request();
      HloInstruction* operand = Visit(outfeed_request.operand(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateOutfeed(
              operand, outfeed_request.outfeed_config()));
      break;
    }

    case OpRequest::kMapRequest: {
      const MapRequest& map_request = request.request().map_request();
      std::vector<HloInstruction*> operands;
      for (const ComputationDataHandle& handle : map_request.operands()) {
        HloInstruction* operand = Visit(handle, visited);
        operands.push_back(operand);
      }
      CHECK_EQ(1, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version map_version =
          request.embedded_computation_versions(0);
      HloComputation* map_computation =
          ResolveComputation(map_request.to_apply(), map_version);
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateMap(
          request.output_shape(), operands, map_computation));
      break;
    }

    case OpRequest::kReduceRequest: {
      const ReduceRequest& reduce_request = request.request().reduce_request();
      HloInstruction* operand = Visit(reduce_request.operand(), visited);
      HloInstruction* init_value = Visit(reduce_request.init_value(), visited);
      CHECK_EQ(1, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version reduce_version =
          request.embedded_computation_versions(0);
      HloComputation* reduce_computation =
          ResolveComputation(reduce_request.to_apply(), reduce_version);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateReduce(
              request.output_shape(), operand, init_value,
              AsInt64Slice(reduce_request.dimensions()), reduce_computation));
      break;
    }

    case OpRequest::kReduceWindowRequest: {
      const ReduceWindowRequest& reduce_window_request =
          request.request().reduce_window_request();
      HloInstruction* operand = Visit(reduce_window_request.operand(), visited);
      HloInstruction* init_value =
          Visit(reduce_window_request.init_value(), visited);
      CHECK_EQ(1, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version reduce_window_version =
          request.embedded_computation_versions(0);
      HloComputation* reduce_window_computation = ResolveComputation(
          reduce_window_request.to_apply(), reduce_window_version);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateReduceWindow(
              request.output_shape(), operand, init_value,
              reduce_window_request.window(), reduce_window_computation));
      break;
    }

    case OpRequest::kSelectAndScatterRequest: {
      const SelectAndScatterRequest& select_and_scatter_request =
          request.request().select_and_scatter_request();
      HloInstruction* operand =
          Visit(select_and_scatter_request.operand(), visited);
      HloInstruction* source =
          Visit(select_and_scatter_request.source(), visited);
      HloInstruction* init_value =
          Visit(select_and_scatter_request.init_value(), visited);
      CHECK_EQ(2, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version select_version =
          request.embedded_computation_versions(0);
      VersionedComputationHandle::Version scatter_version =
          request.embedded_computation_versions(1);
      HloComputation* select_computation = ResolveComputation(
          select_and_scatter_request.select(), select_version);
      HloComputation* scatter_computation = ResolveComputation(
          select_and_scatter_request.scatter(), scatter_version);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateSelectAndScatter(
              request.output_shape(), operand, select_computation,
              select_and_scatter_request.window(), source, init_value,
              scatter_computation));
      break;
    }

    case OpRequest::kBroadcastRequest: {
      const BroadcastRequest& broadcast_request =
          request.request().broadcast_request();
      HloInstruction* operand = Visit(broadcast_request.operand(), visited);
      std::vector<int64> broadcast_dimensions;
      // The client-level broadcast instruction just appends dimensions on the
      // left (adds lowest numbered dimensions). The HLO broadcast op is more
      // flexible and can add new dimensions anywhere. The broadcast_dimensions
      // maps operand dimensions to dimensions in the broadcast output, so
      // to append dimensions on the left the broadcast_dimensions should just
      // be the n highest dimension numbers of the output shape where n is
      // the number of input dimensions.
      for (int i = 0; i < ShapeUtil::Rank(operand->shape()); ++i) {
        broadcast_dimensions.push_back(i +
                                       ShapeUtil::Rank(request.output_shape()) -
                                       ShapeUtil::Rank(operand->shape()));
      }
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateBroadcast(
              request.output_shape(), operand, broadcast_dimensions));
      break;
    }

    case OpRequest::kReshapeRequest: {
      const ReshapeRequest& reshape_request =
          request.request().reshape_request();
      HloInstruction* operand = Visit(reshape_request.operand(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateReshape(
              request.output_shape(),
              hlo_builder_.AddInstruction(HloInstruction::CreateTranspose(
                  ShapeUtil::PermuteDimensions(
                      InversePermutation(
                          AsInt64Slice(reshape_request.dimensions())),
                      operand->shape()),
                  operand, AsInt64Slice(reshape_request.dimensions())))));
      break;
    }

    case OpRequest::kReverseRequest: {
      const ReverseRequest& reverse_request =
          request.request().reverse_request();
      HloInstruction* operand = Visit(reverse_request.operand(), visited);
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateReverse(
              request.output_shape(), operand,
              AsInt64Slice(reverse_request.dimensions())));
      break;
    }

    case OpRequest::kPadRequest: {
      const PadRequest& pad_request = request.request().pad_request();
      HloInstruction* operand = Visit(pad_request.operand(), visited);
      HloInstruction* padding_value =
          Visit(pad_request.padding_value(), visited);
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreatePad(
          request.output_shape(), operand, padding_value,
          pad_request.padding_config()));
      break;
    }

    case OpRequest::kRecvRequest: {
      const RecvRequest& recv_request = request.request().recv_request();
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateRecv(
          request.output_shape(), recv_request.channel_handle().handle()));
      break;
    }

    case OpRequest::kParameterRequest: {
      const ParameterRequest& parameter_request =
          request.request().parameter_request();
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateParameter(
              parameter_request.parameter(), request.output_shape(),
              parameter_request.name()));
      break;
    }

    case OpRequest::kConvertRequest: {
      const ConvertRequest& convert_request =
          request.request().convert_request();
      HloInstruction* operand = Visit(convert_request.operand(), visited);
      hlo_instruction = hlo_builder_.AddInstruction(
          HloInstruction::CreateConvert(request.output_shape(), operand));
      break;
    }

    case OpRequest::kWhileRequest: {
      const WhileRequest& while_request = request.request().while_request();
      CHECK_EQ(2, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version condition_version =
          request.embedded_computation_versions(0);
      HloComputation* condition =
          ResolveComputation(while_request.condition(), condition_version);
      VersionedComputationHandle::Version body_version =
          request.embedded_computation_versions(1);
      HloComputation* body =
          ResolveComputation(while_request.body(), body_version);
      HloInstruction* init = Visit(while_request.init(), visited);
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateWhile(
          request.output_shape(), condition, body, init));
      break;
    }

    case OpRequest::kTernaryOpRequest: {
      const TernaryOpRequest& ternary_op_request =
          request.request().ternary_op_request();
      HloInstruction* lhs = Visit(ternary_op_request.lhs(), visited);
      HloInstruction* rhs = Visit(ternary_op_request.rhs(), visited);
      HloInstruction* ehs = Visit(ternary_op_request.ehs(), visited);
      auto hlo_opcode = TernaryOperationToHloOpcode(ternary_op_request.triop());
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateTernary(
              request.output_shape(), hlo_opcode, lhs, rhs, ehs));
      break;
    }

    case OpRequest::kVariadicOpRequest: {
      const VariadicOpRequest& variadic_op_request =
          request.request().variadic_op_request();
      std::vector<HloInstruction*> operands;
      for (const ComputationDataHandle& handle :
           variadic_op_request.operands()) {
        HloInstruction* operand = Visit(handle, visited);
        operands.push_back(operand);
      }
      auto hlo_opcode =
          VariadicOperationToHloOpcode(variadic_op_request.varop());
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateVariadic(
              request.output_shape(), hlo_opcode, operands));
      break;
    }

    case OpRequest::kCallRequest: {
      const CallRequest& call_request = request.request().call_request();
      std::vector<HloInstruction*> operands;
      for (const ComputationDataHandle& handle : call_request.operands()) {
        operands.push_back(Visit(handle, visited));
      }
      CHECK_EQ(1, request.embedded_computation_versions_size());
      VersionedComputationHandle::Version call_version =
          request.embedded_computation_versions(0);
      HloComputation* call_computation =
          ResolveComputation(call_request.to_apply(), call_version);
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateCall(
          request.output_shape(), operands, call_computation));
      break;
    }

    case OpRequest::kCustomCallRequest: {
      const CustomCallRequest& cc_request =
          request.request().custom_call_request();
      std::vector<HloInstruction*> operands;
      for (const ComputationDataHandle& operand : cc_request.operands()) {
        operands.push_back(Visit(operand, visited));
      }
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateCustomCall(
              cc_request.shape(), operands, cc_request.call_target_name()));
      break;
    }

    case OpRequest::kUnaryOpRequest: {
      const UnaryOpRequest& unary_op_request =
          request.request().unary_op_request();
      HloInstruction* operand = Visit(unary_op_request.operand(), visited);
      auto hlo_opcode = UnaryOperationToHloOpcode(unary_op_request.unop());
      hlo_instruction = hlo_builder_.AddInstruction(HloInstruction::CreateUnary(
          request.output_shape(), hlo_opcode, operand));
      break;
    }

    case OpRequest::kBinaryOpRequest: {
      const BinaryOpRequest& binary_op_request =
          request.request().binary_op_request();
      HloInstruction* lhs = Visit(binary_op_request.lhs(), visited);
      HloInstruction* rhs = Visit(binary_op_request.rhs(), visited);
      auto hlo_opcode = BinaryOperationToHloOpcode(binary_op_request.binop());
      if (binary_op_request.broadcast_dimensions_size() > 0) {
        // Emit a broadcast instruction to perform the "broadcast in dimension"
        // operation.
        CHECK_NE(ShapeUtil::Rank(lhs->shape()), ShapeUtil::Rank(rhs->shape()));
        HloInstruction* operand_to_broadcast =
            ShapeUtil::Rank(lhs->shape()) < ShapeUtil::Rank(rhs->shape()) ? lhs
                                                                          : rhs;
        Shape broadcast_shape = ShapeUtil::MakeShape(
            operand_to_broadcast->shape().element_type(),
            AsInt64Slice(request.output_shape().dimensions()));

        CHECK_EQ(ShapeUtil::Rank(operand_to_broadcast->shape()),
                 binary_op_request.broadcast_dimensions().size());
        // The broadcast semantics of a client-level binary op broadcast is
        // identical to the HLO broadcast semantics so the broadcast_dimensions
        // field can just be passed to the instruction builder.
        HloInstruction* broadcasted_operand =
            hlo_builder_.AddInstruction(HloInstruction::CreateBroadcast(
                broadcast_shape, operand_to_broadcast,
                AsInt64Slice(binary_op_request.broadcast_dimensions())));

        lhs = (lhs == operand_to_broadcast) ? broadcasted_operand : lhs;
        rhs = (rhs == operand_to_broadcast) ? broadcasted_operand : rhs;
      }
      hlo_instruction =
          hlo_builder_.AddInstruction(HloInstruction::CreateBinary(
              request.output_shape(), hlo_opcode, lhs, rhs));
      break;
    }

    case OpRequest::OP_NOT_SET:
      LOG(FATAL) << "OperationRequest doesn't contain a request";

    default:
      LOG(FATAL) << "Unexpected request type: " << request.request().op_case();
  }
  (*visited)[handle.handle()] = hlo_instruction;
  return hlo_instruction;
}

}  // namespace

StatusOr<std::unique_ptr<HloComputation>> UserComputation::BuildHloComputation(
    VersionedComputationHandle::Version version,
    HloComputationResolver hlo_resolver, bool include_unused_parameters) const {
  tensorflow::mutex_lock lock(mutex_);

  VLOG(2) << "Building HloComputation from UserComputation " << name_
          << " at version " << version << ". Operation requests:\n"
          << session_computation_.ShortDebugString();

  std::unique_ptr<HloComputation> hlo_computation = ComputationLowerer::Lower(
      tensorflow::strings::StrCat(name(), ".v", version), session_computation_,
      version, std::move(hlo_resolver), include_unused_parameters);

  VLOG(2) << "HloComputation:\n" << hlo_computation->ToString();
  return std::move(hlo_computation);
}

}  // namespace xla
