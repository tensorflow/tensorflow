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

#include "xla/backends/cpu/runtime/thunk_serdes_proto.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/all_gather_thunk.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/all_to_all_thunk.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/collective_permute_thunk.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/dot_thunk.h"
#include "xla/backends/cpu/runtime/fft_thunk.h"
#include "xla/backends/cpu/runtime/infeed_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/outfeed_thunk.h"
#include "xla/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/serdes_base.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/topk_thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

static absl::StatusOr<ResourceProto> ToProto(const Resource& resource) {
  ResourceProto proto;
  switch (resource.kind()) {
    case Resource::kToken:
      proto.set_kind(ResourceProto::TOKEN);
      break;
    case Resource::kCollectiveCommunicator:
      proto.set_kind(ResourceProto::COLLECTIVE_COMMUNICATOR);
      break;
    default:
      return absl::UnimplementedError("Resource kind not supported.");
  }
  return proto;
}

static InfoProto ThunkInfoToProto(const Thunk::Info& info) {
  InfoProto proto;
  proto.set_op_name(info.op_name);
  proto.set_module_name(info.module_name);
  proto.set_module_id(info.module_id);
  return proto;
}

static absl::StatusOr<OpParamsProto> ToProto(
    const CollectiveThunk::OpParams& op_params) {
  OpParamsProto proto;
  proto.set_has_channel_id(op_params.has_channel_id);
  proto.set_use_global_device_ids(
      op_params.use_global_device_ids.value());  // TODO(basioli) optional
  proto.set_op_id(op_params.op_id);
  for (const auto& group : op_params.group) {
    ReplicaGroup* replica_group = proto.add_replica_group();
    for (const auto& device : group.replica_ids()) {
      replica_group->add_replica_ids(device);
    }
  }
  return proto;
}

static absl::StatusOr<BufferAllocationSliceProto> ToProto(
    const BufferAllocation::Slice& slice) {
  BufferAllocationSliceProto proto;
  proto.set_offset(slice.offset());
  proto.set_size(slice.size());
  proto.set_buffer_allocation_index(
      slice.allocation() == nullptr ? -1 : slice.index());
  return proto;
}

static absl::Status SerializeSliceShapeIntoProto(
    const BufferAllocation::Slice& slice, const Shape& shape,
    ShapeBufferAllocationSliceProto* proto) {
  *proto->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*proto->mutable_slice(), ToProto(slice));
  return absl::OkStatus();
}

class ThunkSerDesProtobuf : public SerDesBase<Thunk> {
  friend class ThunkSequenceSerDesProtobuf;

 public:
  explicit ThunkSerDesProtobuf(const BufferAssignment* buffer_assignment =
                                   nullptr);  // NOTE buffer assignment isn't
                                              // needed for serialization.
  absl::StatusOr<std::string> Serialize(const Thunk& thunk) override;
  absl::StatusOr<std::unique_ptr<Thunk>> Deserialize(
      const std::string& serialized) override;

 protected:
  absl::StatusOr<ThunkProto> ToProto(const Thunk& thunk) const;
  absl::StatusOr<std::unique_ptr<Thunk>> FromProto(
      const ThunkProto& proto) const;

 private:
  // TODO(basiol) remove NOLINT when this actually gets used
  const BufferAssignment* buffer_assignment_;  // NOLINT
};

ThunkSerDesProtobuf::ThunkSerDesProtobuf(
    const BufferAssignment* buffer_assignment)
    : buffer_assignment_(buffer_assignment) {}

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
static absl::Status ToProto(const AllGatherThunk& thunk,
                            AllGatherThunkProto& proto) {
  // NOTE(basioli) AllGatherThunkProto has no extra fields to serialize.
  return absl::OkStatus();
}

static absl::Status ToProto(const AllReduceThunk& thunk,
                            AllReduceThunkProto& proto) {
  absl::string_view reduction_kind_as_string_view =
      ReductionKindToString(thunk.reduction_kind());
  std::string reduction_kind_as_string(reduction_kind_as_string_view.begin(),
                                       reduction_kind_as_string_view.end());
  proto.set_reduction_kind(reduction_kind_as_string);
  proto.set_single_replica(thunk.single_replica());
  return absl::OkStatus();
}

static absl::Status ToProto(const AllToAllThunk& thunk,
                            AllToAllThunkProto& proto) {
  // NOTE(basioli) AllToAllThunkProto has no extra fields to serialize.
  return absl::OkStatus();
}

static absl::Status ToProto(const ReduceScatterThunk& thunk,
                            ReduceScatterThunkProto& proto) {
  absl::string_view reduction_kind_as_string_view =
      ReductionKindToString(thunk.reduction_kind());
  std::string reduction_kind_as_string(reduction_kind_as_string_view.begin(),
                                       reduction_kind_as_string_view.end());
  proto.set_reduction_kind(reduction_kind_as_string);
  return absl::OkStatus();
}

static absl::Status ToProto(const CollectivePermuteThunk& thunk,
                            CollectivePermuteThunkProto& proto) {
  for (const auto& source_target_pair : thunk.source_target_pairs()) {
    CollectivePermuteThunkProto::SourceTargetPairProto*
        source_target_pair_proto = proto.add_source_target_pairs();
    source_target_pair_proto->set_source(source_target_pair.first);
    source_target_pair_proto->set_target(source_target_pair.second);
  }
  return absl::OkStatus();
}

static absl::Status ToProto(const CollectiveThunk& thunk, ThunkProto& proto) {
  CollectiveThunkProto* collective_thunk_proto =
      proto.mutable_collective_thunk();

  TF_ASSIGN_OR_RETURN(*collective_thunk_proto->mutable_op_params(),
                      ToProto(thunk.op_params()));

  TF_ASSIGN_OR_RETURN(*collective_thunk_proto->mutable_op_resources()
                           ->mutable_communicator_resource(),
                      ToProto(*thunk.op_resources().communicator_resource));

  for (size_t i = 0; i < thunk.op_buffers().source_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().source_buffers[i],
        thunk.op_buffers().source_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_source_shapes_buffer_slices()));
  }

  for (size_t i = 0; i < thunk.op_buffers().destination_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().destination_buffers[i],
        thunk.op_buffers().destination_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_destination_shapes_buffer_slices()));
  }

  if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllGather)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllGatherThunk&>(thunk),
                *collective_thunk_proto->mutable_all_gather_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllReduce)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllReduceThunk&>(thunk),
                *collective_thunk_proto->mutable_all_reduce_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllToAll)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllToAllThunk&>(thunk),
                *collective_thunk_proto->mutable_all_to_all_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kReduceScatter)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const ReduceScatterThunk&>(thunk),
                *collective_thunk_proto->mutable_reduce_scatter_thunk()));
  } else if (proto.kind() ==
             Thunk::KindToString(Thunk::Kind::kCollectivePermute)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const CollectivePermuteThunk&>(thunk),
                *collective_thunk_proto->mutable_collective_permute_thunk()));
  } else {
    return absl::UnimplementedError(
        "SerializeAsStringCollectiveImpl not implemented");
  }

  return absl::OkStatus();
}

static absl::Status ToProto(const CallThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  CallThunkProto* call_thunk_proto = proto.mutable_call_thunk();

  TF_ASSIGN_OR_RETURN(
      *call_thunk_proto->mutable_called_sequence(),
      thunk_sequence_serdes.ToProto(thunk.called_executor().thunk_sequence()));
  return absl::OkStatus();
}

static absl::Status ToProto(const CopyThunk& thunk, ThunkProto& proto) {
  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.src_buffer(), thunk.src_shape(),
      copy_thunk_proto->mutable_src_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dst_buffer(), thunk.dst_shape(),
      copy_thunk_proto->mutable_dst_buffer_shape()));
  return absl::OkStatus();
}

static absl::Status ToProto(const CustomCallThunk& thunk, ThunkProto& proto) {
  CustomCallThunkProto* custom_call_thunk_proto =
      proto.mutable_custom_call_thunk();

  custom_call_thunk_proto->set_target_name(thunk.target_name());
  custom_call_thunk_proto->set_backend_config(thunk.backend_config());
  custom_call_thunk_proto->set_api_version(thunk.api_version());

  for (size_t i = 0; i < thunk.op_buffers().arguments_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().arguments_buffers[i],
        thunk.op_buffers().arguments_shapes[i],
        custom_call_thunk_proto->mutable_op_buffers()->add_arguments_shapes()));
  }

  for (size_t i = 0; i < thunk.op_buffers().results_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().results_buffers[i],
        thunk.op_buffers().results_shapes[i],
        custom_call_thunk_proto->mutable_op_buffers()->add_results_shapes()));
  }

  return absl::OkStatus();
}

static absl::Status ToProto(const ConvolutionThunk& thunk, ThunkProto& proto) {
  ConvolutionThunkProto* convolution_thunk_proto =
      proto.mutable_convolution_thunk();

  const std::string dnums_as_str = thunk.dnums().SerializeAsString();
  convolution_thunk_proto->mutable_dimension_numbers()->ParseFromString(
      dnums_as_str);

  const std::string window_as_str = thunk.window().SerializeAsString();
  convolution_thunk_proto->mutable_window()->ParseFromString(window_as_str);

  convolution_thunk_proto->set_feature_group_count(thunk.feature_group_count());

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.input_buffer(), thunk.input_shape(),
      convolution_thunk_proto->mutable_input_buffer_shape()));

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.output_buffer(), thunk.output_shape(),
      convolution_thunk_proto->mutable_output_buffer_shape()));

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.kernel_buffer(), thunk.kernel_shape(),
      convolution_thunk_proto->mutable_kernel_buffer_shape()));

  convolution_thunk_proto->mutable_options()->set_multi_threaded(
      thunk.options().multi_threaded);
  convolution_thunk_proto->mutable_options()->set_use_acl(
      thunk.options().use_acl);

  return absl::OkStatus();
}

static absl::Status ToProto(const DotThunk& thunk, ThunkProto& proto) {
  DotThunkProto* dot_thunk_proto = proto.mutable_dot_thunk();

  *dot_thunk_proto->mutable_dot_dimensions() = thunk.dot_dimensions();
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().lhs_buffer, thunk.dot_slices().lhs_shape,
      dot_thunk_proto->mutable_lhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().rhs_buffer, thunk.dot_slices().rhs_shape,
      dot_thunk_proto->mutable_rhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().out_buffer, thunk.dot_slices().out_shape,
      dot_thunk_proto->mutable_out_buffer_shape()));

  return absl::OkStatus();
}

static absl::Status ToProto(const InfeedThunk& thunk, ThunkProto& proto) {
  InfeedThunkProto* infeed_thunk_proto = proto.mutable_infeed_thunk();

  TF_ASSIGN_OR_RETURN(
      *infeed_thunk_proto->mutable_infeed_resources()->mutable_consume_token(),
      ToProto(*thunk.infeed_resources().consume_token));
  TF_ASSIGN_OR_RETURN(
      *infeed_thunk_proto->mutable_infeed_resources()->mutable_produce_token(),
      ToProto(*thunk.infeed_resources().produce_token));

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
  TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                           ->mutable_consume_token(),
                      ToProto(*thunk.outfeed_resources().consume_token));
  TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                           ->mutable_produce_token(),
                      ToProto(*thunk.outfeed_resources().produce_token));

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
  switch (thunk.direction()) {
    case SortThunk::SortDirection::kAscending:
      sort_thunk_proto->set_direction(SortThunkProto::ASCENDING);
      break;
    case SortThunk::SortDirection::kDescending:
      sort_thunk_proto->set_direction(SortThunkProto::DESCENDING);
      break;
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
                      ToProto(thunk.values_buffer()));
  TF_ASSIGN_OR_RETURN(*top_k_thunk_proto->mutable_indices_buffer(),
                      ToProto(thunk.indices_buffer()));
  TF_ASSIGN_OR_RETURN(*top_k_thunk_proto->mutable_output_buffer(),
                      ToProto(thunk.output_buffer()));
  return absl::OkStatus();
}

static absl::Status ToProto(const WhileThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  WhileThunkProto* while_thunk_proto = proto.mutable_while_thunk();
  while_thunk_proto->set_trip_count(thunk.trip_count().value());

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_cond_sequence(),
      thunk_sequence_serdes.ToProto(thunk.cond_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_body_sequence(),
      thunk_sequence_serdes.ToProto(thunk.body_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(*while_thunk_proto->mutable_cond_buffer(),
                      ToProto(thunk.cond_buffer()));
  return absl::OkStatus();
}

static absl::Status ToProto(const XnnDotThunk& thunk, ThunkProto& proto) {
  XnnDotThunkProto* xnn_dot_thunk_proto = proto.mutable_xnn_dot_thunk();
  *xnn_dot_thunk_proto->mutable_dot_dimensions() = thunk.dot_dimensions();
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().lhs_buffer, thunk.dot_slices().lhs_shape,
      xnn_dot_thunk_proto->mutable_lhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().rhs_buffer, thunk.dot_slices().rhs_shape,
      xnn_dot_thunk_proto->mutable_rhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dot_slices().out_buffer, thunk.dot_slices().out_shape,
      xnn_dot_thunk_proto->mutable_out_buffer_shape()));
  return absl::OkStatus();
}

static absl::Status ToProto(const FftThunk& thunk, ThunkProto& proto) {
  FftThunkProto* fft_thunk_proto = proto.mutable_fft_thunk();

  fft_thunk_proto->set_is_multi_thread_eigen(thunk.is_multi_thread_eigen());
  fft_thunk_proto->set_fft_type(thunk.fft_type());
  const auto& fft_length = thunk.fft_length();
  fft_thunk_proto->mutable_fft_length()->Add(fft_length.begin(),
                                             fft_length.end());

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.input_buffer(), thunk.input_shape(),
      fft_thunk_proto->mutable_input_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.output_buffer(), thunk.output_shape(),
      fft_thunk_proto->mutable_output_buffer_shape()));

  return absl::OkStatus();
}

static absl::Status ToProto(const RngGetAndUpdateStateThunk& thunk,
                            ThunkProto& proto) {
  RngGetAndUpdateStateThunkProto* rng_get_and_update_state_thunk_proto =
      proto.mutable_rng_get_and_update_state_thunk();

  rng_get_and_update_state_thunk_proto->set_delta(thunk.delta());

  TF_ASSIGN_OR_RETURN(
      *rng_get_and_update_state_thunk_proto->mutable_state_buffer(),
      ToProto(thunk.state_buffer()));

  return absl::OkStatus();
}

static absl::Status ToProto(const KernelThunk& thunk, ThunkProto& proto) {
  KernelThunkProto* kernel_thunk_proto = proto.mutable_kernel_thunk();

  kernel_thunk_proto->set_kernel_name(thunk.kernel_name());
  kernel_thunk_proto->mutable_thread_dim()->set_x(thunk.thread_dim().x);
  kernel_thunk_proto->mutable_thread_dim()->set_y(thunk.thread_dim().y);
  kernel_thunk_proto->mutable_thread_dim()->set_z(thunk.thread_dim().z);
  kernel_thunk_proto->set_min_alignment(thunk.min_alignment().value());

  for (const BufferAllocation::Slice& buffer : thunk.arguments_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_arguments_buffers(),
                        ToProto(buffer));
  }

  for (const BufferAllocation::Slice& buffer : thunk.results_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_results_buffers(),
                        ToProto(buffer));
  }
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
                      ToProto(thunk.branch_index_buffer()));
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
  switch (thunk.kind()) {
    // NOTE collective thunks
    case Thunk::Kind::kAllGather:
    case Thunk::Kind::kAllReduce:
    case Thunk::Kind::kAllToAll:
    case Thunk::Kind::kCollectivePermute:
    case Thunk::Kind::kReduceScatter:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const CollectiveThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kConditional:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const ConditionalThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kFft:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const FftThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kRngGetAndUpdateState:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const RngGetAndUpdateStateThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kKernel:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const KernelThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kCall:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const CallThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kCopy:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const CopyThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kCustomCall:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const CustomCallThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kConvolution:
      TF_RETURN_IF_ERROR(::xla::cpu::ToProto(
          static_cast<const ConvolutionThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kDot:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const DotThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kInfeed:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const InfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kOutfeed:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const OutfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kSort:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const SortThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kTopK:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const TopKThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kWhile:
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const WhileThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kXnnFusion:
      // TODO(basioli) follow up CL
      // should add XnnDotThunk, we should abide by the same inheritance
      // pattern as the other thunks.
      TF_RETURN_IF_ERROR(
          ::xla::cpu::ToProto(static_cast<const XnnDotThunk&>(thunk), proto));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("ToProto is not implemented for thunk kind: %s",
                          Thunk::KindToString(thunk.kind())));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<Thunk>> ThunkSerDesProtobuf::FromProto(
    const ThunkProto& proto) const {
  // big switch case here
  return absl::UnimplementedError("FromProto is not implemented");
}

ThunkSequenceSerDesProtobuf::ThunkSequenceSerDesProtobuf(
    const BufferAssignment* buffer_assignment)
    : buffer_assignment_(buffer_assignment) {}

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
  ThunkSerDesProtobuf thunk_serdes(buffer_assignment_);
  ThunkSequenceProto proto;
  proto.mutable_thunks()->Reserve(thunk_sequence.size());
  for (auto& thunk : thunk_sequence) {
    TF_ASSIGN_OR_RETURN(*proto.add_thunks(), thunk_serdes.ToProto(*thunk));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<ThunkSequence>>
ThunkSequenceSerDesProtobuf::FromProto(const ThunkSequenceProto& proto) const {
  ThunkSerDesProtobuf thunk_serdes(buffer_assignment_);
  auto thunk_sequence = std::make_unique<ThunkSequence>();
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                        thunk_serdes.FromProto(thunk_proto));
    thunk_sequence->push_back(std::move(thunk));
  }
  return thunk_sequence;
}

}  // namespace xla::cpu
