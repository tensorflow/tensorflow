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

#include "xla/backends/gpu/transforms/estimate_cub_scan_scratch_size.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Invokes the FFI instantiate handler to compute the scratch buffer size.
static absl::StatusOr<int64_t> InvokeInstantiateHandlerAndGetScratchSize(
    const ffi::HandlerRegistration& registration, ffi::CallFrame call_frame) {
  ffi::ExecutionState state;
  ffi::InvokeContext context{};
  context.state_context = {&state, nullptr, nullptr};
  RETURN_IF_ERROR(ffi::Invoke(ffi::GetXlaFfiApi(),
                              registration.bundle.instantiate, call_frame,
                              context, XLA_FFI_ExecutionStage_INSTANTIATE));

  // Read the scratch size from the execution state.
  ASSIGN_OR_RETURN(int64_t* scratch_size_ptr, state.Get<int64_t>());
  return *scratch_size_ptr;
}

absl::Status EstimateCubScanScratchSize::RunOnScanInstruction(
    HloCustomCallInstruction* custom_call) {
  CHECK_EQ(custom_call->custom_call_target(),
           kCubDeviceScanUnassignedScratchSizeTarget);
  ASSIGN_OR_RETURN(CubScanOptions options,
                   custom_call->backend_config<CubScanOptions>());

  ASSIGN_OR_RETURN(ffi::HandlerRegistration handler,
                   ffi::FindHandler(kCubDeviceScanUnassignedScratchSizeTarget,
                                    platform_name_));

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  xla::PrimitiveType type = custom_call->operand(0)->shape().element_type();
  attrs.Insert("element_type", static_cast<int32_t>(type));
  attrs.Insert("vector_length", options.vector_length());
  attrs.Insert("row_length", options.row_length());
  attrs.Insert("column_length", options.column_length());
  attrs.Insert("kind", static_cast<int32_t>(options.kind()));
  attrs.Insert("is_reverse", options.is_reverse());

  ffi::CallFrameBuilder builder(0, 0);
  builder.AddAttributes(attrs.Build());

  ASSIGN_OR_RETURN(
      int64_t scratch_size,
      InvokeInstantiateHandlerAndGetScratchSize(handler, builder.Build()));

  // Replace the custom call with one that has a scratch size.
  Shape new_shape = custom_call->shape();
  new_shape.mutable_tuple_shapes()->back() =
      ShapeUtil::MakeShape(U8, {scratch_size});
  HloInstruction* new_custom_call =
      custom_call->AddInstruction(HloInstruction::CreateCustomCall(
          new_shape, custom_call->operands(), kCubDeviceScanTarget));
  static_cast<HloCustomCallInstruction*>(new_custom_call)
      ->set_api_version(CustomCallApiVersion::API_VERSION_TYPED_FFI);
  std::string backend_config = absl::StrFormat(
      "{vector_length = %d : i64, row_length = %d : i64, "
      "column_length = %d : i64, kind = %d : i32, is_reverse = %s}",
      options.vector_length(), options.row_length(), options.column_length(),
      static_cast<int32_t>(options.kind()),
      options.is_reverse() ? "true" : "false");
  new_custom_call->set_raw_backend_config_string(backend_config);
  RETURN_IF_ERROR(custom_call->parent()->ReplaceInstructionWithDifferentShape(
      custom_call, new_custom_call));
  return absl::OkStatus();
}

absl::StatusOr<bool> EstimateCubScanScratchSize::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloCustomCallInstruction*> custom_calls;
  for (auto* inst : computation->instructions()) {
    if (auto custom_call = DynCast<HloCustomCallInstruction>(inst)) {
      if (custom_call->custom_call_target() ==
          kCubDeviceScanUnassignedScratchSizeTarget) {
        custom_calls.push_back(custom_call);
      }
    }
  }
  bool changed = false;
  for (auto* call : custom_calls) {
    RETURN_IF_ERROR(RunOnScanInstruction(call));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> EstimateCubScanScratchSize::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "EstimateCubScanScratchSize::RunImpl(), before:\n" +
                        module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(3, "EstimateCubScanScratchSize::RunImpl(), after:\n" +
                        module->ToString());
  return changed;
}

}  // namespace xla::gpu
