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

#include "xla/backends/cpu/runtime/thunk_serdes/fft_thunk_serdes.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/fft_thunk.h"
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

absl::Status FftThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& fft_thunk = absl::down_cast<const FftThunk&>(thunk);
  FftThunkProto* fft_thunk_proto = proto.mutable_fft_thunk();

  fft_thunk_proto->set_is_multi_thread_eigen(fft_thunk.is_multi_thread_eigen());
  fft_thunk_proto->set_fft_type(fft_thunk.fft_type());
  const auto& fft_length = fft_thunk.fft_length();
  fft_thunk_proto->mutable_fft_length()->Add(fft_length.begin(),
                                             fft_length.end());

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      fft_thunk.input_buffer(), fft_thunk.input_shape(),
      fft_thunk_proto->mutable_input_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      fft_thunk.output_buffer(), fft_thunk.output_shape(),
      fft_thunk_proto->mutable_output_buffer_shape()));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> FftThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto input_slice_shape,
      DeserializeSliceShapeFromProto(proto.fft_thunk().input_buffer_shape(),
                                     buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto output_slice_shape,
      DeserializeSliceShapeFromProto(proto.fft_thunk().output_buffer_shape(),
                                     buffer_allocations));

  const auto& [input_buffer, input_shape] = input_slice_shape;
  const auto& [output_buffer, output_shape] = output_slice_shape;

  return FftThunk::Create(
      std::move(info), proto.fft_thunk().is_multi_thread_eigen(),
      proto.fft_thunk().fft_type(), proto.fft_thunk().fft_length(),
      input_buffer, input_shape, output_buffer, output_shape);
}

}  // namespace

void RegisterFftThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(
      Thunk::Kind::kFft, FftThunkToProto, FftThunkFromProto));
}

// Statically registers the FftThunk serialization/deserialization logic.
static bool fft_thunk_serdes_registered = [] {
  RegisterFftThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
