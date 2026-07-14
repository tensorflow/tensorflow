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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_UTILS_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_UTILS_H_

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

namespace xla::cpu {

absl::Status SerializeSliceShapeIntoProto(
    const BufferAllocation::Slice& slice, const Shape& shape,
    ShapeBufferAllocationSliceProto* proto);

absl::StatusOr<std::pair<BufferAllocation::Slice, Shape>>
DeserializeSliceShapeFromProto(
    const ShapeBufferAllocationSliceProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations);

InfoProto ThunkInfoToProto(const Thunk::Info& info);

absl::StatusOr<Thunk::Info> ThunkInfoFromProto(const InfoProto& proto);

absl::StatusOr<std::shared_ptr<Resource>> CreateResourceFromProto(
    const ResourceProto& proto);

absl::StatusOr<ResourceProto> ToProto(const Resource& resource);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_UTILS_H_
