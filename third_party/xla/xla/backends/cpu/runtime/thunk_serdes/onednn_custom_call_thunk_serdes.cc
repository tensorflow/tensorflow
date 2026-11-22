/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/thunk_serdes/onednn_custom_call_thunk_serdes.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/onednn_config.pb.h"

namespace xla::cpu {
namespace {

using OneDnnOpConfig = OneDnnOpThunk::OneDnnOpConfig;

// oneDNN custom call targets that the thunk runtime knows how to execute.
constexpr char kOneDnnMatmulTarget[] = "__onednn$matmul";
constexpr char kOneDnnConvolutionTarget[] = "__onednn$convolution";
constexpr char kOneDnnLayerNormTarget[] = "__onednn$layernorm";
constexpr char kOneDnnSoftmaxTarget[] = "__onednn$softmax";

// Every oneDNN op supported by the thunk runtime adds an alternative to
// `OneDnnOpConfig`. Fail the build when a new one is added so that serdes
// support (and a round trip test) is added for it here as well.
static_assert(std::variant_size_v<OneDnnOpConfig> == 4,
              "A new oneDNN op was added to OneDnnOpThunk::OneDnnOpConfig. "
              "Add serialization support for it in OneDnnCustomCallToProto and "
              "OneDnnCustomCallFromProto, then update this count.");

// Serializes the buffer slices, shapes and the tuple-ness of the result, all of
// which are laid out identically for every oneDNN target.
absl::Status SerializeOneDnnOpBuffersIntoProto(
    const OneDnnOpThunk::OpBuffers& op_buffers,
    OneDnnOpBuffersProto* op_buffers_proto) {
  // The structured binding fails to compile if a field is added to `OpBuffers`,
  // which makes sure the new field to be serialized here and deserialized in
  // `DeserializeOneDnnOpBuffersFromProto`.
  const auto& [arguments_buffers, arguments_shapes, results_buffers,
               results_shapes, is_tuple_result] = op_buffers;

  if (arguments_buffers.size() != arguments_shapes.size() ||
      results_buffers.size() != results_shapes.size()) {
    return absl::InvalidArgumentError(
        "Number of oneDNN buffer slices does not match the number of shapes");
  }

  for (size_t i = 0; i < arguments_buffers.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        SerializeSliceShapeIntoProto(arguments_buffers[i], arguments_shapes[i],
                                     op_buffers_proto->add_arguments_shapes()));
  }

  for (size_t i = 0; i < results_buffers.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        SerializeSliceShapeIntoProto(results_buffers[i], results_shapes[i],
                                     op_buffers_proto->add_results_shapes()));
  }

  op_buffers_proto->set_is_tuple_result(is_tuple_result);
  return absl::OkStatus();
}

absl::StatusOr<OneDnnOpThunk::OpBuffers> DeserializeOneDnnOpBuffersFromProto(
    const OneDnnOpBuffersProto& op_buffers_proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  OneDnnOpThunk::OpBuffers op_buffers;

  for (const ShapeBufferAllocationSliceProto& arg_buff_shape :
       op_buffers_proto.arguments_shapes()) {
    ABSL_ASSIGN_OR_RETURN(
        auto args_slice_shape,
        DeserializeSliceShapeFromProto(arg_buff_shape, buffer_allocations));

    const auto& [args_buffer, args_shape] = args_slice_shape;
    op_buffers.arguments_buffers.push_back(args_buffer);
    op_buffers.arguments_shapes.push_back(args_shape);
  }

  for (const ShapeBufferAllocationSliceProto& res_buff_shape :
       op_buffers_proto.results_shapes()) {
    ABSL_ASSIGN_OR_RETURN(
        auto res_slice_shape,
        DeserializeSliceShapeFromProto(res_buff_shape, buffer_allocations));

    const auto& [res_buffer, res_shape] = res_slice_shape;
    op_buffers.results_buffers.push_back(res_buffer);
    op_buffers.results_shapes.push_back(res_shape);
  }

  op_buffers.is_tuple_result = op_buffers_proto.is_tuple_result();
  return op_buffers;
}

// Serializes the target specific config together with the op buffers into the
// proto message of a single oneDNN target.
template <typename ConfigType, typename OneDnnTargetProto>
absl::Status SerializeOneDnnTargetIntoProto(const OneDnnOpThunk& thunk,
                                            OneDnnTargetProto* target_proto) {
  const ConfigType* config = std::get_if<ConfigType>(&thunk.onednn_op_config());
  if (config == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing oneDNN config in backend config for target: ",
                     thunk.target()));
  }
  *target_proto->mutable_config() = *config;
  return SerializeOneDnnOpBuffersIntoProto(thunk.op_buffers(),
                                           target_proto->mutable_op_buffers());
}

absl::Status OneDnnCustomCallToProto(const OneDnnOpThunk& thunk,
                                     OneDnnCustomCallProto& onednn_proto) {
  const std::string& target = thunk.target();

  if (target == kOneDnnMatmulTarget) {
    return SerializeOneDnnTargetIntoProto<OneDnnMatMulConfig>(
        thunk, onednn_proto.mutable_onednn_matmul_thunk());
  }
  if (target == kOneDnnConvolutionTarget) {
    return SerializeOneDnnTargetIntoProto<OneDnnConvolutionConfig>(
        thunk, onednn_proto.mutable_onednn_convolution_thunk());
  }
  if (target == kOneDnnLayerNormTarget) {
    return SerializeOneDnnTargetIntoProto<OneDnnNormConfig>(
        thunk, onednn_proto.mutable_onednn_layernorm_thunk());
  }
  if (target == kOneDnnSoftmaxTarget) {
    return SerializeOneDnnTargetIntoProto<OneDnnSoftmaxConfig>(
        thunk, onednn_proto.mutable_onednn_softmax_thunk());
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported OneDNN operation target: ", target));
}

}  // namespace

std::optional<absl::Status> TrySerializeOneDnnCustomCall(const Thunk& thunk,
                                                         ThunkProto& proto) {
  const auto* onednn_thunk = dynamic_cast<const OneDnnOpThunk*>(&thunk);
  if (onednn_thunk == nullptr) {
    return std::nullopt;
  }
  return OneDnnCustomCallToProto(*onednn_thunk,
                                 *proto.mutable_onednn_custom_call_thunk());
}

absl::StatusOr<std::unique_ptr<Thunk>> OneDnnCustomCallFromProto(
    const OneDnnCustomCallProto& onednn_proto, Thunk::Info info,
    const std::vector<BufferAllocation>& buffer_allocations) {
  std::string target;
  OneDnnOpConfig onednn_op_config;
  const OneDnnOpBuffersProto* op_buffers_proto = nullptr;

  switch (onednn_proto.impl_case()) {
    case OneDnnCustomCallProto::kOnednnMatmulThunk:
      target = kOneDnnMatmulTarget;
      onednn_op_config = onednn_proto.onednn_matmul_thunk().config();
      op_buffers_proto = &onednn_proto.onednn_matmul_thunk().op_buffers();
      break;
    case OneDnnCustomCallProto::kOnednnConvolutionThunk:
      target = kOneDnnConvolutionTarget;
      onednn_op_config = onednn_proto.onednn_convolution_thunk().config();
      op_buffers_proto = &onednn_proto.onednn_convolution_thunk().op_buffers();
      break;
    case OneDnnCustomCallProto::kOnednnLayernormThunk:
      target = kOneDnnLayerNormTarget;
      onednn_op_config = onednn_proto.onednn_layernorm_thunk().config();
      op_buffers_proto = &onednn_proto.onednn_layernorm_thunk().op_buffers();
      break;
    case OneDnnCustomCallProto::kOnednnSoftmaxThunk:
      target = kOneDnnSoftmaxTarget;
      onednn_op_config = onednn_proto.onednn_softmax_thunk().config();
      op_buffers_proto = &onednn_proto.onednn_softmax_thunk().op_buffers();
      break;
    case OneDnnCustomCallProto::IMPL_NOT_SET:
      return absl::InvalidArgumentError("Unknown OneDNN operation in proto");
  }

  ABSL_ASSIGN_OR_RETURN(OneDnnOpThunk::OpBuffers op_buffers,
                   DeserializeOneDnnOpBuffersFromProto(*op_buffers_proto,
                                                       buffer_allocations));

  // The serialized `op_name` already carries the target suffix added by
  // `OneDnnOpThunk::Create`, so restore `info` verbatim.
  return OneDnnOpThunk::CreateFromSerialized(target, std::move(info),
                                             std::move(op_buffers),
                                             std::move(onednn_op_config));
}

}  // namespace xla::cpu
