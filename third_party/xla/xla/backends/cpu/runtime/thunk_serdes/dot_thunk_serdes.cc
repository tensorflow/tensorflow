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

#include "xla/backends/cpu/runtime/thunk_serdes/dot_thunk_serdes.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/dot_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

absl::Status DotThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& dot_thunk = absl::down_cast<const DotThunk&>(thunk);
  DotThunkProto* dot_thunk_proto = proto.mutable_dot_thunk();

  *dot_thunk_proto->mutable_dot_dimensions() = dot_thunk.dot_dimensions();
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      dot_thunk.dot_slices().lhs_buffer, dot_thunk.dot_slices().lhs_shape,
      dot_thunk_proto->mutable_lhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      dot_thunk.dot_slices().rhs_buffer, dot_thunk.dot_slices().rhs_shape,
      dot_thunk_proto->mutable_rhs_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      dot_thunk.dot_slices().out_buffer, dot_thunk.dot_slices().out_shape,
      dot_thunk_proto->mutable_out_buffer_shape()));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> DotThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto lhs_slice_shape,
      DeserializeSliceShapeFromProto(proto.dot_thunk().lhs_buffer_shape(),
                                     buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto rhs_slice_shape,
      DeserializeSliceShapeFromProto(proto.dot_thunk().rhs_buffer_shape(),
                                     buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto out_slice_shape,
      DeserializeSliceShapeFromProto(proto.dot_thunk().out_buffer_shape(),
                                     buffer_allocations));

  const auto& [lhs_buffer, lhs_shape] = lhs_slice_shape;
  const auto& [rhs_buffer, rhs_shape] = rhs_slice_shape;
  const auto& [out_buffer, out_shape] = out_slice_shape;

  return DotThunk::Create(std::move(info), proto.dot_thunk().dot_dimensions(),
                          std::move(lhs_buffer), lhs_shape,
                          std::move(rhs_buffer), rhs_shape,
                          std::move(out_buffer), out_shape);
}

}  // namespace

void RegisterDotThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(
      Thunk::Kind::kDot, DotThunkToProto, DotThunkFromProto));
}

// Statically register the DotThunk serialization/deserialization logic.
static bool dot_thunk_serdes_registered = [] {
  RegisterDotThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
