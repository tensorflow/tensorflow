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

#include "xla/backends/cpu/runtime/thunk_serdes/copy_thunk_serdes.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
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

static absl::Status CopyThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& copy_thunk = tsl::down_cast<const CopyThunk&>(thunk);
  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      copy_thunk.src_buffer(), copy_thunk.src_shape(),
      copy_thunk_proto->mutable_src_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      copy_thunk.dst_buffer(), copy_thunk.dst_shape(),
      copy_thunk_proto->mutable_dst_buffer_shape()));
  return absl::OkStatus();
}

static absl::StatusOr<std::unique_ptr<Thunk>> CopyThunkFromProto(
    const ThunkProto& proto, const std::vector<BufferAllocation>& allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(auto src_slice_shape,
                      DeserializeSliceShapeFromProto(
                          proto.copy_thunk().src_buffer_shape(), allocations));
  TF_ASSIGN_OR_RETURN(auto dst_slice_shape,
                      DeserializeSliceShapeFromProto(
                          proto.copy_thunk().dst_buffer_shape(), allocations));

  const auto& [src_buffer, src_shape] = src_slice_shape;
  const auto& [dst_buffer, dst_shape] = dst_slice_shape;

  return CopyThunk::Create(std::move(info), std::move(src_buffer), src_shape,
                           std::move(dst_buffer), dst_shape);
}

void RegisterCopyThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(
      Thunk::Kind::kCopy, CopyThunkToProto, CopyThunkFromProto));
}

// Statically registers the CopyThunk serialization/deserialization logic.
static bool custom_call_thunk_serdes_registered = [] {
  RegisterCopyThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
