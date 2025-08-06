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

#include "xla/stream_executor/launch_dim.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

absl::StatusOr<Dim3D> Dim3D::FromProto(const Dim3DProto& proto) {
  if (proto.x() <= 0 || proto.y() <= 0 || proto.z() <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Launch dimensions need to be positive integers (x=%v, y=%v, z=%v).",
        proto.x(), proto.y(), proto.z()));
  }
  return Dim3D{static_cast<uint64_t>(proto.x()),
               static_cast<uint64_t>(proto.y()),
               static_cast<uint64_t>(proto.z())};
}

Dim3DProto stream_executor::Dim3D::ToProto() const {
  Dim3DProto proto;
  proto.set_x(x);
  proto.set_y(y);
  proto.set_z(z);
  return proto;
}

ThreadDimProto stream_executor::ThreadDim::ToProto() const {
  ThreadDimProto proto;
  *proto.mutable_coordinates() = Dim3D::ToProto();
  return proto;
}

absl::StatusOr<ThreadDim> ThreadDim::FromProto(const ThreadDimProto& proto) {
  TF_ASSIGN_OR_RETURN(Dim3D coordinates, Dim3D::FromProto(proto.coordinates()));
  return ThreadDim(coordinates);
}

BlockDimProto stream_executor::BlockDim::ToProto() const {
  BlockDimProto proto;
  *proto.mutable_coordinates() = Dim3D::ToProto();
  return proto;
}

absl::StatusOr<BlockDim> BlockDim::FromProto(const BlockDimProto& proto) {
  TF_ASSIGN_OR_RETURN(Dim3D coordinates, Dim3D::FromProto(proto.coordinates()));
  return BlockDim(coordinates);
}

ClusterDimProto stream_executor::ClusterDim::ToProto() const {
  ClusterDimProto proto;
  *proto.mutable_coordinates() = Dim3D::ToProto();
  return proto;
}

absl::StatusOr<ClusterDim> ClusterDim::FromProto(const ClusterDimProto& proto) {
  TF_ASSIGN_OR_RETURN(Dim3D coordinates, Dim3D::FromProto(proto.coordinates()));
  return ClusterDim(coordinates);
}

}  // namespace stream_executor
