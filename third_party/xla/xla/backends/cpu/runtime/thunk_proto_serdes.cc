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

#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"

#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/infeed_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/logical_id_thunk.h"
#include "xla/backends/cpu/runtime/outfeed_thunk.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/serdes_base.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/backends/cpu/runtime/topk_thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/resource_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

ThunkSerDesRegistry& ThunkSerDesRegistry::Get() {
  static auto* registry = new ThunkSerDesRegistry();
  return *registry;
}

absl::Status ThunkSerDesRegistry::Register(Thunk::Kind kind, ToProtoFn to_proto,
                                           FromProtoFn from_proto) {
  if (to_proto_fns_.contains(kind)) {
    return Internal("ToProto function already registered for thunk kind: %s",
                    Thunk::KindToString(kind));
  }
  to_proto_fns_[kind] = std::move(to_proto);
  if (from_proto_fns_.contains(kind)) {
    return Internal("FromProto function already registered for thunk kind: %s",
                    Thunk::KindToString(kind));
  }
  from_proto_fns_[kind] = std::move(from_proto);
  return absl::OkStatus();
}

absl::StatusOr<ThunkSerDesRegistry::ToProtoFn>
ThunkSerDesRegistry::GetToProtoFn(Thunk::Kind kind) const {
  auto it = to_proto_fns_.find(kind);
  if (it == to_proto_fns_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No ToProto function registered for thunk kind: %s",
                        Thunk::KindToString(kind)));
  }
  return it->second;
}

absl::StatusOr<ThunkSerDesRegistry::FromProtoFn>
ThunkSerDesRegistry::GetFromProtoFn(Thunk::Kind kind) const {
  auto it = from_proto_fns_.find(kind);
  if (it == from_proto_fns_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No FromProto function registered for thunk kind: %s",
                        Thunk::KindToString(kind)));
  }
  return it->second;
}

void ForEachThunkProto(const ThunkSequenceProto& proto,
                       std::function<void(const ThunkProto&)> callback) {
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    if (thunk_proto.has_call_thunk()) {
      ForEachThunkProto(thunk_proto.call_thunk().called_sequence(), callback);
    } else if (thunk_proto.has_conditional_thunk()) {
      for (const ThunkSequenceProto& branch_sequence :
           thunk_proto.conditional_thunk().branch_sequences()) {
        ForEachThunkProto(branch_sequence, callback);
      }
    } else if (thunk_proto.has_while_thunk()) {
      ForEachThunkProto(thunk_proto.while_thunk().body_sequence(), callback);
      ForEachThunkProto(thunk_proto.while_thunk().cond_sequence(), callback);
    }
    callback(thunk_proto);
  }
}

static absl::StatusOr<Thunk::Kind> ProtoThunkToThunkKind(
    const ThunkProto& proto) {
  switch (proto.impl_case()) {
    case ThunkProto::ImplCase::kCollectiveThunk:
      return Thunk::Kind::kCollective;
    case ThunkProto::ImplCase::kCallThunk:
      return Thunk::Kind::kCall;
    case ThunkProto::ImplCase::kConditionalThunk:
      return Thunk::Kind::kConditional;
    case ThunkProto::ImplCase::kConvolutionThunk:
      return Thunk::Kind::kConvolution;
    case ThunkProto::ImplCase::kCopyThunk:
      return Thunk::Kind::kCopy;
    case ThunkProto::ImplCase::kCustomCallThunk:
      return Thunk::Kind::kCustomCall;
    case ThunkProto::ImplCase::kDotThunk:
      return Thunk::Kind::kDot;
    case ThunkProto::ImplCase::kFftThunk:
      return Thunk::Kind::kFft;
    case ThunkProto::ImplCase::kInfeedThunk:
      return Thunk::Kind::kInfeed;
    case ThunkProto::ImplCase::kKernelThunk:
      return Thunk::Kind::kKernel;
    case ThunkProto::ImplCase::kOutfeedThunk:
      return Thunk::Kind::kOutfeed;
    case ThunkProto::ImplCase::kRngGetAndUpdateStateThunk:
      return Thunk::Kind::kRngGetAndUpdateState;
    case ThunkProto::ImplCase::kSortThunk:
      return Thunk::Kind::kSort;
    case ThunkProto::ImplCase::kTopKThunk:
      return Thunk::Kind::kTopK;
    case ThunkProto::ImplCase::kWhileThunk:
      return Thunk::Kind::kWhile;
    case ThunkProto::ImplCase::kXnnFusionThunk:
      return Internal("Thunk kind kXnnFusionThunk is deprecated.");
    case ThunkProto::ImplCase::kPartitionIdThunk:
      return Thunk::Kind::kPartitionId;
    case ThunkProto::ImplCase::kReplicaIdThunk:
      return Thunk::Kind::kReplicaId;
    case ThunkProto::ImplCase::kYnnFusionThunk:
      return Thunk::Kind::kYnnFusion;
    case ThunkProto::ImplCase::IMPL_NOT_SET:
      return Internal("Thunk kind not set.");
  }
}

class ThunkSerDesProtobuf : public SerDesBase<Thunk> {
  friend class ThunkSequenceSerDesProtobuf;

 public:
  // Buffer allocations and resources are not needed for serialization.
  explicit ThunkSerDesProtobuf(
      const HloModule* hlo_module = nullptr,
      const std::vector<BufferAllocation>* buffer_allocations = nullptr,
      const std::vector<std::shared_ptr<Resource>>* thunk_resources = nullptr);
  absl::StatusOr<std::string> Serialize(const Thunk& thunk) override;
  absl::StatusOr<std::unique_ptr<Thunk>> Deserialize(
      const std::string& serialized) override;

 protected:
  absl::StatusOr<ThunkProto> ToProto(const Thunk& thunk) const;
  absl::StatusOr<std::unique_ptr<Thunk>> FromProto(
      const ThunkProto& proto) const;

 private:
  const HloModule* hlo_module_;
  const std::vector<BufferAllocation>* buffer_allocations_;

  const std::vector<std::shared_ptr<Resource>>* thunk_resources_;
};

ThunkSerDesProtobuf::ThunkSerDesProtobuf(
    const HloModule* hlo_module,
    const std::vector<BufferAllocation>* buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>* thunk_resources)
    : hlo_module_(hlo_module),
      buffer_allocations_(buffer_allocations),
      thunk_resources_(thunk_resources) {}

absl::StatusOr<std::string> ThunkSerDesProtobuf::Serialize(const Thunk& thunk) {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, ToProto(thunk));
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<Thunk>> ThunkSerDesProtobuf::Deserialize(
    const std::string& serialized) {
  ThunkProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse thunk proto:\n %s", serialized));
  }
  return FromProto(proto);
}

static absl::Status ToProto(const CallThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  CallThunkProto* call_thunk_proto = proto.mutable_call_thunk();

  TF_ASSIGN_OR_RETURN(
      *call_thunk_proto->mutable_called_sequence(),
      thunk_sequence_serdes.ToProto(thunk.called_executor().thunk_sequence()));
  return absl::OkStatus();
}

static absl::Status ToProto(const InfeedThunk& thunk, ThunkProto& proto) {
  InfeedThunkProto* infeed_thunk_proto = proto.mutable_infeed_thunk();

  infeed_thunk_proto->mutable_infeed_resources()
      ->mutable_consume_token()
      ->set_contains_value(thunk.infeed_resources().consume_token != nullptr);
  if (thunk.infeed_resources().consume_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*infeed_thunk_proto->mutable_infeed_resources()
                             ->mutable_consume_token()
                             ->mutable_value(),
                        ToProto(*thunk.infeed_resources().consume_token));
  }

  infeed_thunk_proto->mutable_infeed_resources()
      ->mutable_produce_token()
      ->set_contains_value(thunk.infeed_resources().produce_token != nullptr);
  if (thunk.infeed_resources().produce_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*infeed_thunk_proto->mutable_infeed_resources()
                             ->mutable_produce_token()
                             ->mutable_value(),
                        ToProto(*thunk.infeed_resources().produce_token));
  }

  for (const InfeedThunk::InfeedBuffer& infeed_buffer :
       thunk.infeed_buffers()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        infeed_buffer.slice, infeed_buffer.shape,
        infeed_thunk_proto->add_infeed_buffers_shapes()));
  }
  return absl::OkStatus();
}

static absl::Status ToProto(const OutfeedThunk& thunk, ThunkProto& proto) {
  OutfeedThunkProto* outfeed_thunk_proto = proto.mutable_outfeed_thunk();
  outfeed_thunk_proto->mutable_outfeed_resources()
      ->mutable_consume_token()
      ->set_contains_value(thunk.outfeed_resources().consume_token != nullptr);
  if (thunk.outfeed_resources().consume_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                             ->mutable_consume_token()
                             ->mutable_value(),
                        ToProto(*thunk.outfeed_resources().consume_token));
  }

  outfeed_thunk_proto->mutable_outfeed_resources()
      ->mutable_produce_token()
      ->set_contains_value(thunk.outfeed_resources().produce_token != nullptr);
  if (thunk.outfeed_resources().produce_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                             ->mutable_produce_token()
                             ->mutable_value(),
                        ToProto(*thunk.outfeed_resources().produce_token));
  }

  for (const OutfeedThunk::OutfeedBuffer& outfeed_buffer :
       thunk.outfeed_buffers()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        outfeed_buffer.slice, outfeed_buffer.shape,
        outfeed_thunk_proto->add_outfeed_buffers_shapes()));
  }
  return absl::OkStatus();
}

static absl::Status ToProto(const SortThunk& thunk, ThunkProto& proto) {
  SortThunkProto* sort_thunk_proto = proto.mutable_sort_thunk();

  sort_thunk_proto->set_dimension(thunk.dimension());
  sort_thunk_proto->set_is_stable(thunk.is_stable());
  sort_thunk_proto->set_comparator_name(thunk.comparator_name());
  sort_thunk_proto->mutable_direction()->set_contains_value(
      thunk.direction().has_value());
  if (thunk.direction().has_value()) {
    switch (*thunk.direction()) {
      case SortThunk::SortDirection::kAscending:
        sort_thunk_proto->mutable_direction()->set_value(
            SortDirectionProto::ASCENDING);
        break;
      case SortThunk::SortDirection::kDescending:
        sort_thunk_proto->mutable_direction()->set_value(
            SortDirectionProto::DESCENDING);
        break;
    }
  }

  if (thunk.has_less_than()) {
    return absl::UnimplementedError("`LessThan` is not serializable.");
  }

  for (const SortThunk::Input& input : thunk.inputs()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        input.slice, input.shape, sort_thunk_proto->add_inputs_shapes()));
  }

  return absl::OkStatus();
}

static absl::Status ToProto(const TopKThunk& thunk, ThunkProto& proto) {
  TopKThunkProto* top_k_thunk_proto = proto.mutable_top_k_thunk();
  top_k_thunk_proto->set_batch_size(thunk.batch_size());
  top_k_thunk_proto->set_input_size(thunk.input_size());
  top_k_thunk_proto->set_k(thunk.k());

  TF_ASSIGN_OR_RETURN(*top_k_thunk_proto->mutable_values_buffer(),
                      thunk.values_buffer().ToProto());
  TF_ASSIGN_OR_RETURN(*top_k_thunk_proto->mutable_indices_buffer(),
                      thunk.indices_buffer().ToProto());
  TF_ASSIGN_OR_RETURN(*top_k_thunk_proto->mutable_output_buffer(),
                      thunk.output_buffer().ToProto());
  return absl::OkStatus();
}

static absl::Status ToProto(const WhileThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  WhileThunkProto* while_thunk_proto = proto.mutable_while_thunk();
  while_thunk_proto->mutable_trip_count()->set_contains_value(
      thunk.trip_count().has_value());
  if (thunk.trip_count().has_value()) {
    while_thunk_proto->mutable_trip_count()->set_value(*thunk.trip_count());
  }

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_cond_sequence(),
      thunk_sequence_serdes.ToProto(thunk.cond_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_body_sequence(),
      thunk_sequence_serdes.ToProto(thunk.body_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(*while_thunk_proto->mutable_cond_buffer(),
                      thunk.cond_buffer().ToProto());
  return absl::OkStatus();
}

static absl::Status ToProto(const RngGetAndUpdateStateThunk& thunk,
                            ThunkProto& proto) {
  RngGetAndUpdateStateThunkProto* rng_get_and_update_state_thunk_proto =
      proto.mutable_rng_get_and_update_state_thunk();

  rng_get_and_update_state_thunk_proto->set_delta(thunk.delta());

  TF_ASSIGN_OR_RETURN(
      *rng_get_and_update_state_thunk_proto->mutable_state_buffer(),
      thunk.state_buffer().ToProto());

  return absl::OkStatus();
}

static absl::Status ToProto(const KernelThunkBase& thunk, ThunkProto& proto) {
  KernelThunkProto* kernel_thunk_proto = proto.mutable_kernel_thunk();

  // NOTE OSS doesn't accept string_view as a parameter to set_kernel_name
  const absl::string_view kernel_name = thunk.kernel_name();
  const std::string kernel_name_str(kernel_name.begin(), kernel_name.end());
  kernel_thunk_proto->set_kernel_name(kernel_name_str);
  kernel_thunk_proto->mutable_num_workgroups()->set_x(thunk.num_workgroups().x);
  kernel_thunk_proto->mutable_num_workgroups()->set_y(thunk.num_workgroups().y);
  kernel_thunk_proto->mutable_num_workgroups()->set_z(thunk.num_workgroups().z);
  kernel_thunk_proto->mutable_min_alignment()->set_contains_value(
      thunk.min_alignment().has_value());
  if (thunk.min_alignment().has_value()) {
    kernel_thunk_proto->mutable_min_alignment()->set_value(
        *thunk.min_alignment());
  }

  for (const ShapedSlice& buffer : thunk.arguments_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_arguments_buffers(),
                        buffer.ToProto());
  }

  for (const ShapedSlice& buffer : thunk.results_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_results_buffers(),
                        buffer.ToProto());
  }

  std::vector<int64_t> invariant_arguments(thunk.invariant_arguments().begin(),
                                           thunk.invariant_arguments().end());
  absl::c_sort(invariant_arguments);
  kernel_thunk_proto->mutable_invariant_arguments()->Add(
      invariant_arguments.begin(), invariant_arguments.end());

  return absl::OkStatus();
}

static absl::Status ToProto(const ConditionalThunk& thunk, ThunkProto& proto) {
  ConditionalThunkProto* conditional_thunk_proto =
      proto.mutable_conditional_thunk();
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;

  conditional_thunk_proto->mutable_branch_sequences()->Reserve(
      thunk.branch_executors().size());
  for (const auto& branch_executor : thunk.branch_executors()) {
    TF_ASSIGN_OR_RETURN(
        *conditional_thunk_proto->add_branch_sequences(),
        thunk_sequence_serdes.ToProto(branch_executor.thunk_sequence()));
  }

  TF_ASSIGN_OR_RETURN(*conditional_thunk_proto->mutable_branch_index_buffer(),
                      thunk.branch_index_buffer().ToProto());
  return absl::OkStatus();
}

static absl::Status ToProto(const PartitionIdThunk& thunk, ThunkProto& proto) {
  TF_ASSIGN_OR_RETURN(
      *proto.mutable_partition_id_thunk()->mutable_logical_id_buffer(),
      thunk.logical_id_buffer().ToProto());
  return absl::OkStatus();
}

static absl::Status ToProto(const ReplicaIdThunk& thunk, ThunkProto& proto) {
  TF_ASSIGN_OR_RETURN(
      *proto.mutable_replica_id_thunk()->mutable_logical_id_buffer(),
      thunk.logical_id_buffer().ToProto());
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> ThunkSerDesProtobuf::ToProto(
    const Thunk& thunk) const {
  ThunkProto proto;
  // NOTE OSS doesn't accept string_view as a parameter to set_kind
  const auto kind_as_str_view = Thunk::KindToString(thunk.kind());
  const std::string kind_as_str(kind_as_str_view);
  proto.set_kind(kind_as_str);
  *proto.mutable_info() = ThunkInfoToProto(thunk.info());

  // Check if there is a registered ToProto function for this thunk kind.
  auto to_proto_fn_or = ThunkSerDesRegistry::Get().GetToProtoFn(thunk.kind());
  if (to_proto_fn_or.ok()) {
    TF_RETURN_IF_ERROR((*to_proto_fn_or)(thunk, proto));
    return proto;
  }

  switch (thunk.kind()) {
    case Thunk::Kind::kConditional:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const ConditionalThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kRngGetAndUpdateState:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const RngGetAndUpdateStateThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kKernel:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const KernelThunkBase&>(thunk), proto));
      break;
    case Thunk::Kind::kCall:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(absl::down_cast<const CallThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kInfeed:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const InfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kOutfeed:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const OutfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kSort:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(absl::down_cast<const SortThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kTopK:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(absl::down_cast<const TopKThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kWhile:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          absl::down_cast<const WhileThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kPartitionId:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const PartitionIdThunk&>(
              absl::down_cast<const internal::LogicalIdThunk<
                  internal::LogicalIdKind::kPartitionId>&>(thunk)),
          proto));
      break;
    case Thunk::Kind::kReplicaId:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const ReplicaIdThunk&>(
              absl::down_cast<const internal::LogicalIdThunk<
                  internal::LogicalIdKind::kReplicaId>&>(thunk)),
          proto));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("ToProto is not implemented for thunk kind: %s",
                          Thunk::KindToString(thunk.kind())));
  }
  return proto;
}

static absl::StatusOr<std::unique_ptr<CallThunk>> CallThunkFromProto(
    const ThunkProto& proto, const HloModule* hlo_module,
    const std::vector<BufferAllocation>* buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(hlo_module,
                                                    buffer_allocations);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> call_sequence,
      thunk_sequence_serdes.FromProto(proto.call_thunk().called_sequence()));
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  return CallThunk::Create(std::move(info), std::move(*call_sequence));
}

static absl::StatusOr<std::unique_ptr<ConditionalThunk>>
ConditionalThunkFromProto(
    const ThunkProto& proto, const HloModule* hlo_module,
    const std::vector<BufferAllocation>* buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(hlo_module,
                                                    buffer_allocations);

  std::vector<ThunkSequence> branch_sequences;
  for (const ThunkSequenceProto& branch_sequence_proto :
       proto.conditional_thunk().branch_sequences()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ThunkSequence> branch_sequence,
                        thunk_sequence_serdes.FromProto(branch_sequence_proto));
    branch_sequences.push_back(std::move(*branch_sequence));
  }
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice branch_index_buffer,
                      BufferAllocation::Slice::FromProto(
                          proto.conditional_thunk().branch_index_buffer(),
                          *buffer_allocations));

  return ConditionalThunk::Create(std::move(info),
                                  std::move(branch_index_buffer),
                                  std::move(branch_sequences));
}

static absl::StatusOr<std::unique_ptr<InfeedThunk>> InfeedThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<InfeedThunk::InfeedBuffer> infeed_buffers;
  for (const ShapeBufferAllocationSliceProto& infeed_buffer_shape :
       proto.infeed_thunk().infeed_buffers_shapes()) {
    TF_ASSIGN_OR_RETURN(auto infeed_buffer_slice_shape,
                        DeserializeSliceShapeFromProto(infeed_buffer_shape,
                                                       buffer_allocations));

    const auto& [infeed_buffer, infeed_shape] = infeed_buffer_slice_shape;
    infeed_buffers.push_back(
        {std::move(infeed_buffer), std::move(infeed_shape)});
  }

  InfeedThunk::InfeedResources infeed_resources;
  if (proto.infeed_thunk()
          .infeed_resources()
          .consume_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        infeed_resources.consume_token,
        CreateResourceFromProto(
            proto.infeed_thunk().infeed_resources().consume_token().value()));
  } else {
    infeed_resources.consume_token = nullptr;
  }

  if (proto.infeed_thunk()
          .infeed_resources()
          .produce_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        infeed_resources.produce_token,
        CreateResourceFromProto(
            proto.infeed_thunk().infeed_resources().produce_token().value()));
  } else {
    infeed_resources.produce_token = nullptr;
  }

  return InfeedThunk::Create(std::move(info), std::move(infeed_buffers),
                             std::move(infeed_resources));
}

static absl::StatusOr<std::unique_ptr<Thunk>> KernelThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<ShapedSlice> arguments_buffers;
  std::vector<ShapedSlice> results_buffers;

  for (const ShapedSliceProto& buffer_proto :
       proto.kernel_thunk().arguments_buffers()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer, ShapedSlice::FromProto(buffer_proto, buffer_allocations));
    arguments_buffers.push_back(std::move(buffer));
  }

  for (const ShapedSliceProto& buffer_proto :
       proto.kernel_thunk().results_buffers()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer, ShapedSlice::FromProto(buffer_proto, buffer_allocations));
    results_buffers.push_back(std::move(buffer));
  }

  NumWorkGroups num_workgroups{
      static_cast<uint64_t>(proto.kernel_thunk().num_workgroups().x()),
      static_cast<uint64_t>(proto.kernel_thunk().num_workgroups().y()),
      static_cast<uint64_t>(proto.kernel_thunk().num_workgroups().z())};

  absl::flat_hash_set<int64_t> invariant_arguments;
  for (int64_t invariant_argument :
       proto.kernel_thunk().invariant_arguments()) {
    invariant_arguments.insert(invariant_argument);
  }

  std::optional<uint64_t> min_alignment = std::nullopt;
  if (proto.kernel_thunk().min_alignment().contains_value()) {
    min_alignment = proto.kernel_thunk().min_alignment().value();
  }

  return KernelThunk::Create(std::move(info), std::move(arguments_buffers),
                             std::move(results_buffers),
                             proto.kernel_thunk().kernel_name(), num_workgroups,
                             invariant_arguments, min_alignment);
}

static absl::StatusOr<std::unique_ptr<OutfeedThunk>> OutfeedThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<OutfeedThunk::OutfeedBuffer> outfeed_buffers;
  for (const ShapeBufferAllocationSliceProto& buffer_proto :
       proto.outfeed_thunk().outfeed_buffers_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer_slice_shape,
        DeserializeSliceShapeFromProto(buffer_proto, buffer_allocations));

    const auto& [buffer_slice, buffer_shape] = buffer_slice_shape;
    outfeed_buffers.push_back(
        {std::move(buffer_slice), std::move(buffer_shape)});
  }

  OutfeedThunk::OutfeedResources outfeed_resources;
  if (proto.outfeed_thunk()
          .outfeed_resources()
          .consume_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        outfeed_resources.consume_token,
        CreateResourceFromProto(
            proto.outfeed_thunk().outfeed_resources().consume_token().value()));
  } else {
    outfeed_resources.consume_token = nullptr;
  }

  if (proto.outfeed_thunk()
          .outfeed_resources()
          .produce_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        outfeed_resources.produce_token,
        CreateResourceFromProto(
            proto.outfeed_thunk().outfeed_resources().produce_token().value()));
  } else {
    outfeed_resources.produce_token = nullptr;
  }

  return OutfeedThunk::Create(std::move(info), outfeed_buffers,
                              outfeed_resources);
}

static absl::StatusOr<std::unique_ptr<RngGetAndUpdateStateThunk>>
RngGetAndUpdateStateThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice state_buffer,
                      BufferAllocation::Slice::FromProto(
                          proto.rng_get_and_update_state_thunk().state_buffer(),
                          buffer_allocations));

  return RngGetAndUpdateStateThunk::Create(
      std::move(info), state_buffer,
      proto.rng_get_and_update_state_thunk().delta());
}

static absl::StatusOr<std::unique_ptr<SortThunk>> SortThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));
  std::vector<SortThunk::Input> inputs;
  for (const ShapeBufferAllocationSliceProto& buffer_proto :
       proto.sort_thunk().inputs_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer_slice_shape,
        DeserializeSliceShapeFromProto(buffer_proto, buffer_allocations));

    const auto& [buffer_slice, buffer_shape] = buffer_slice_shape;
    inputs.push_back({std::move(buffer_slice), std::move(buffer_shape)});
  }

  std::optional<SortThunk::SortDirection> sort_direction = std::nullopt;
  if (proto.sort_thunk().direction().contains_value()) {
    sort_direction =
        proto.sort_thunk().direction().value() == SortDirectionProto::ASCENDING
            ? SortThunk::SortDirection::kAscending
            : SortThunk::SortDirection::kDescending;
  }

  return SortThunk::Create(
      std::move(info), inputs, proto.sort_thunk().dimension(),
      proto.sort_thunk().is_stable(), proto.sort_thunk().comparator_name(),
      sort_direction);
}

static absl::StatusOr<std::unique_ptr<TopKThunk>> TopKThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice values_buffer,
      BufferAllocation::Slice::FromProto(proto.top_k_thunk().values_buffer(),
                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice output_buffer,
      BufferAllocation::Slice::FromProto(proto.top_k_thunk().output_buffer(),
                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice indices_buffer,
      BufferAllocation::Slice::FromProto(proto.top_k_thunk().indices_buffer(),
                                         buffer_allocations));

  return TopKThunk::Create(std::move(info), values_buffer, output_buffer,
                           indices_buffer, proto.top_k_thunk().batch_size(),
                           proto.top_k_thunk().input_size(),
                           proto.top_k_thunk().k());
}

static absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunkFromProto(
    const ThunkProto& proto, const HloModule* hlo_module,
    const std::vector<BufferAllocation>* buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(hlo_module,
                                                    buffer_allocations);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> cond_sequence,
      thunk_sequence_serdes.FromProto(proto.while_thunk().cond_sequence()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> body_sequence,
      thunk_sequence_serdes.FromProto(proto.while_thunk().body_sequence()));

  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice cond_buffer,
      BufferAllocation::Slice::FromProto(proto.while_thunk().cond_buffer(),
                                         *buffer_allocations));

  std::optional<int64_t> trip_count = std::nullopt;
  if (proto.while_thunk().trip_count().contains_value()) {
    trip_count = proto.while_thunk().trip_count().value();
  }

  return WhileThunk::Create(std::move(info), cond_buffer,
                            std::move(*cond_sequence),
                            std::move(*body_sequence), trip_count);
}

static absl::StatusOr<std::unique_ptr<Thunk>> PartitionIdThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice logical_id_buffer,
      BufferAllocation::Slice::FromProto(
          proto.partition_id_thunk().logical_id_buffer(), buffer_allocations));

  return internal::LogicalIdThunk<
      internal::LogicalIdKind::kPartitionId>::Create(std::move(info),
                                                     std::move(
                                                         logical_id_buffer));
}

static absl::StatusOr<std::unique_ptr<Thunk>> ReplicaIdThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice logical_id_buffer,
      BufferAllocation::Slice::FromProto(
          proto.replica_id_thunk().logical_id_buffer(), buffer_allocations));

  return internal::LogicalIdThunk<internal::LogicalIdKind::kReplicaId>::Create(
      std::move(info), std::move(logical_id_buffer));
}

absl::StatusOr<std::unique_ptr<Thunk>> ThunkSerDesProtobuf::FromProto(
    const ThunkProto& proto) const {
  CHECK(buffer_allocations_ != nullptr);
  CHECK(thunk_resources_ != nullptr);
  TF_ASSIGN_OR_RETURN(Thunk::Kind kind, ProtoThunkToThunkKind(proto));
  if (Thunk::KindToString(kind) != proto.kind()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Kind mismatch between proto kind `%s` and thunk kind `%s`.",
            proto.kind(), Thunk::KindToString(kind)));
  }

  auto from_proto_fn_or = ThunkSerDesRegistry::Get().GetFromProtoFn(kind);
  if (from_proto_fn_or.ok()) {
    return (*from_proto_fn_or)(proto, *buffer_allocations_, hlo_module_,
                               thunk_resources_);
  }

  switch (kind) {
    case Thunk::Kind::kCall:
      return CallThunkFromProto(proto, hlo_module_, buffer_allocations_);
    case Thunk::Kind::kConditional:
      return ConditionalThunkFromProto(proto, hlo_module_, buffer_allocations_);
    case Thunk::Kind::kInfeed:
      return InfeedThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kKernel:
      return KernelThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kOutfeed:
      return OutfeedThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kRngGetAndUpdateState:
      return RngGetAndUpdateStateThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kSort:
      return SortThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kTopK:
      return TopKThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kWhile:
      return WhileThunkFromProto(proto, hlo_module_, buffer_allocations_);
    case Thunk::Kind::kPartitionId:
      return PartitionIdThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kReplicaId:
      return ReplicaIdThunkFromProto(proto, *buffer_allocations_);
    default:
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          absl::StrFormat("Unsupported thunk kind: %s",
                                          Thunk::KindToString(kind)));
  }
  return absl::UnimplementedError("FromProto is not implemented");
}

ThunkSequenceSerDesProtobuf::ThunkSequenceSerDesProtobuf(
    const HloModule* hlo_module,
    const std::vector<BufferAllocation>* buffer_allocations)
    : hlo_module_(hlo_module), buffer_allocations_(buffer_allocations) {}

absl::StatusOr<std::string> ThunkSequenceSerDesProtobuf::Serialize(
    const ThunkSequence& thunk_sequence) {
  TF_ASSIGN_OR_RETURN(ThunkSequenceProto proto, ToProto(thunk_sequence));
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<ThunkSequence>>
ThunkSequenceSerDesProtobuf::Deserialize(const std::string& serialized) {
  ThunkSequenceProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InternalError(absl::StrFormat(
        "Failed to parse thunk sequence proto:\n %s", serialized));
  }
  return FromProto(proto);
}

absl::StatusOr<ThunkSequenceProto> ThunkSequenceSerDesProtobuf::ToProto(
    const ThunkSequence& thunk_sequence) const {
  ThunkSerDesProtobuf thunk_serdes(hlo_module_, buffer_allocations_);
  ThunkSequenceProto proto;
  proto.mutable_thunks()->Reserve(thunk_sequence.size());

  size_t thunk_index = 0;
  absl::flat_hash_map<Resource*, std::vector<size_t>> resource_users;
  for (auto& thunk : thunk_sequence) {
    TF_ASSIGN_OR_RETURN(*proto.add_thunks(), thunk_serdes.ToProto(*thunk));
    for (auto& resource_use : thunk->resource_uses()) {
      Resource* resource = resource_use.resource().get();
      if (resource) {
        resource_users[resource].push_back(thunk_index);
      }
    }
    thunk_index++;
  }

  for (const auto& [resource, users] : resource_users) {
    ThunkSequenceProto::ResourceUsersProto* resource_users_proto =
        proto.add_thunk_resources();

    switch (resource->kind()) {
      case Resource::Kind::kToken:
        resource_users_proto->mutable_resource()->set_kind(
            ResourceProto::TOKEN);
        break;
      case Resource::Kind::kCollectiveCommunicator:
        resource_users_proto->mutable_resource()->set_kind(
            ResourceProto::COLLECTIVE_COMMUNICATOR);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported resource kind: %d", resource->kind()));
    }

    for (size_t user : users) {
      resource_users_proto->add_thunk_indices(user);
    }
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<ThunkSequence>>
ThunkSequenceSerDesProtobuf::FromProto(const ThunkSequenceProto& proto) const {
  auto thunk_sequence = std::make_unique<ThunkSequence>();

  // For every thunk we store a list of resources that are used by the thunk.
  std::vector<std::vector<std::shared_ptr<Resource>>> thunk_resources;
  thunk_resources.resize(proto.thunks_size());

  for (const auto& resource_users_proto : proto.thunk_resources()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<Resource> resource,
        CreateResourceFromProto(resource_users_proto.resource()));
    for (size_t user : resource_users_proto.thunk_indices()) {
      thunk_resources[user].push_back(resource);
    }
  }

  size_t thunk_index = 0;
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    ThunkSerDesProtobuf thunk_serdes(hlo_module_, buffer_allocations_,
                                     &thunk_resources[thunk_index++]);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                        thunk_serdes.FromProto(thunk_proto));
    thunk_sequence->push_back(std::move(thunk));
  }
  return thunk_sequence;
}

}  // namespace xla::cpu
