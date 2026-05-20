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

#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::Status CreateCubSortCustomCall(HloCustomCallInstruction* custom_call,
                                     int64_t scratch_size,
                                     absl::string_view ffi_target,
                                     bool descending, int64_t batch_size) {
  Shape new_shape = custom_call->shape();
  new_shape.mutable_tuple_shapes()->back() =
      ShapeUtil::MakeShape(U8, {scratch_size});
  HloInstruction* new_custom_call =
      custom_call->AddInstruction(HloInstruction::CreateCustomCall(
          new_shape, absl::MakeSpan(custom_call->operands()), ffi_target));
  auto* new_cc = Cast<HloCustomCallInstruction>(new_custom_call);
  new_cc->set_api_version(CustomCallApiVersion::API_VERSION_TYPED_FFI);
  std::string backend_config =
      absl::StrFormat("{descending = %s, batch_size = %d : i64}",
                      descending ? "true" : "false", batch_size);
  new_custom_call->set_raw_backend_config_string(backend_config);
  new_custom_call->SetupDerivedInstruction(custom_call);
  RETURN_IF_ERROR(custom_call->parent()->ReplaceInstructionWithDifferentShape(
      custom_call, new_custom_call));
  return absl::OkStatus();
}

}  // namespace xla::gpu
