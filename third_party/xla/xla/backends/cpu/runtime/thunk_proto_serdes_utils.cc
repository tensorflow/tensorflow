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

#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::Status SerializeSliceShapeIntoProto(
    const BufferAllocation::Slice& slice, const Shape& shape,
    ShapeBufferAllocationSliceProto* proto) {
  *proto->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*proto->mutable_slice(), slice.ToProto());
  return absl::OkStatus();
}

absl::StatusOr<std::pair<BufferAllocation::Slice, Shape>>
DeserializeSliceShapeFromProto(
    const ShapeBufferAllocationSliceProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice,
      BufferAllocation::Slice::FromProto(proto.slice(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  return std::make_pair(slice, shape);
}

InfoProto ThunkInfoToProto(const Thunk::Info& info) {
  InfoProto proto;
  proto.set_op_name(info.op_name);
  proto.set_module_name(info.module_name);
  proto.set_module_id(info.module_id);
  return proto;
}

absl::StatusOr<Thunk::Info> ThunkInfoFromProto(const InfoProto& proto) {
  Thunk::Info info;
  info.op_name = proto.op_name();
  info.module_name = proto.module_name();
  info.module_id = proto.module_id();
  return info;
}

absl::StatusOr<std::shared_ptr<Resource>> CreateResourceFromProto(
    const ResourceProto& proto) {
  switch (proto.kind()) {
    case ResourceProto::TOKEN:
      return Resource::Create(Resource::kToken);
    case ResourceProto::COLLECTIVE_COMMUNICATOR:
      return Resource::Create(Resource::kCollectiveCommunicator);
    default:
      return absl::UnimplementedError("Resource kind not supported.");
  }
}

absl::StatusOr<ResourceProto> ToProto(const Resource& resource) {
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

}  // namespace xla::cpu
