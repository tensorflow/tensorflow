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

#include "xla/backends/gpu/transforms/estimate_cub_scratch_size.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

// Invokes the FFI instantiate handler to compute the scratch buffer size.
// Builds a call frame with placeholder buffers (nullptr data, correct types
// and dimensions) matching the HLO custom call layout.
static absl::StatusOr<int64_t> ComputeFfiScratchSize(
    const ffi::HandlerRegistration& registration,
    const HloCustomCallInstruction* custom_call, int64_t batch_size) {
  bool is_pairs = custom_call->operand_count() == 2;
  int num_args = is_pairs ? 2 : 1;  // keys [+ values]
  int num_rets = is_pairs ? 3 : 2;  // keys_out [+ values_out] + scratch

  ffi::CallFrameBuilder builder(num_args, num_rets);

  // Keys input placeholder.
  const Shape& key_shape = custom_call->operand(0)->shape();
  int64_t num_items = Product(key_shape.dimensions());
  auto key_bytes =
      primitive_util::ByteWidth(key_shape.element_type()) * num_items;
  se::DeviceAddressBase keys_placeholder(nullptr, key_bytes);
  builder.AddBufferArg(keys_placeholder, key_shape.element_type(),
                       key_shape.dimensions());

  if (is_pairs) {
    // Values input placeholder.
    const Shape& value_shape = custom_call->operand(1)->shape();
    auto value_bytes =
        primitive_util::ByteWidth(value_shape.element_type()) * num_items;
    se::DeviceAddressBase values_placeholder(nullptr, value_bytes);
    builder.AddBufferArg(values_placeholder, value_shape.element_type(),
                         value_shape.dimensions());
  }

  // Keys output placeholder.
  builder.AddBufferRet(keys_placeholder, key_shape.element_type(),
                       key_shape.dimensions());

  if (is_pairs) {
    // Values output placeholder.
    const Shape& value_shape = custom_call->operand(1)->shape();
    auto value_bytes =
        primitive_util::ByteWidth(value_shape.element_type()) * num_items;
    se::DeviceAddressBase values_placeholder(nullptr, value_bytes);
    builder.AddBufferRet(values_placeholder, value_shape.element_type(),
                         value_shape.dimensions());
  }

  // Scratch output placeholder (size 0, we're computing it).
  se::DeviceAddressBase scratch_placeholder(nullptr, 0);
  builder.AddBufferRet(scratch_placeholder, U8, {0});

  // Add attributes matching the MLIR backend config.
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("descending", false);
  attrs.Insert("batch_size", batch_size);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  // Invoke the instantiate handler to compute scratch size.
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

absl::Status EstimateCubScratchSize::RunOnSortInstruction(
    HloCustomCallInstruction* custom_call) {
  CHECK_EQ(custom_call->custom_call_target(),
           kCubDeviceRadixSortUnassignedScratchSizeTarget);

  const Shape& key_shape = custom_call->operand(0)->shape();
  bool is_pairs = custom_call->operand_count() == 2;

  // Read sort direction from SortOptions backend config.
  ASSIGN_OR_RETURN(SortOptions sort_options,
                   custom_call->backend_config<SortOptions>());

  // Determine FFI handler target name.
  std::string ffi_target =
      is_pairs ? "xla.gpu.ext.cub_sort_pairs" : "xla.gpu.ext.cub_sort_keys";

  // Look up the registered FFI handler.
  ASSIGN_OR_RETURN(ffi::HandlerRegistration registration,
                   ffi::FindHandler(ffi_target, platform_name_));

  int64_t num_items = Product(key_shape.dimensions());
  int64_t batch_size = num_items / key_shape.dimensions().back();

  ASSIGN_OR_RETURN(
      int64_t scratch_size,
      ComputeFfiScratchSize(registration, custom_call, batch_size));

  // Create the FFI custom call with correct scratch size and MLIR dict
  // backend config for the FFI handler attributes.
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
                      sort_options.descending() ? "true" : "false", batch_size);
  new_custom_call->set_raw_backend_config_string(backend_config);
  new_custom_call->SetupDerivedInstruction(custom_call);
  RETURN_IF_ERROR(custom_call->parent()->ReplaceInstructionWithDifferentShape(
      custom_call, new_custom_call));
  return absl::OkStatus();
}

// Rewrites a single scan instruction with a custom call.
absl::Status EstimateCubScratchSize::RunOnScanInstruction(
    HloCustomCallInstruction* custom_call) {
  CHECK_EQ(custom_call->custom_call_target(),
           kCubDeviceScanUnassignedScratchSizeTarget);
  TF_ASSIGN_OR_RETURN(CubScanOptions options,
                      custom_call->backend_config<CubScanOptions>());

  TF_ASSIGN_OR_RETURN(ffi::HandlerRegistration handler,
                      ffi::FindHandler("xla.gpu.cub.scan", platform_name_));

  ffi::CallFrameBuilder builder(0, 0);
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  int64_t scratch_size = 0;
  attrs.Insert("temp_bytes", absl::bit_cast<int64_t>(&scratch_size));
  attrs.Insert("vector_length", options.vector_length());
  attrs.Insert("row_length", options.row_length());
  attrs.Insert("column_length", options.column_length());
  attrs.Insert("kind", static_cast<int32_t>(options.kind()));
  attrs.Insert("is_reverse", options.is_reverse());
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  TF_RETURN_IF_ERROR(ffi::Invoke(ffi::GetXlaFfiApi(), handler.bundle.initialize,
                                 call_frame, ffi::InvokeContext{},
                                 XLA_FFI_ExecutionStage_INITIALIZE));

  // Update the custom call.
  Shape new_shape = custom_call->shape();
  new_shape.mutable_tuple_shapes()->back() =
      ShapeUtil::MakeShape(U8, {static_cast<int64_t>(scratch_size)});
  HloInstruction* new_custom_call =
      custom_call->AddInstruction(HloInstruction::CreateCustomCall(
          new_shape, absl::MakeSpan(custom_call->operands()),
          "xla.gpu.ext.cub_scan"));
  static_cast<HloCustomCallInstruction*>(new_custom_call)
      ->set_api_version(CustomCallApiVersion::API_VERSION_TYPED_FFI);
  new_custom_call->SetupDerivedInstruction(custom_call);
  RETURN_IF_ERROR(custom_call->parent()->ReplaceInstructionWithDifferentShape(
      custom_call, new_custom_call));
  return absl::OkStatus();
}

// Rewrites the sorts and scans in the given computation into calls to CUB.
absl::StatusOr<bool> EstimateCubScratchSize::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloCustomCallInstruction*> custom_calls;
  for (auto* inst : computation->instructions()) {
    if (auto custom_call = DynCast<HloCustomCallInstruction>(inst)) {
      if (custom_call->custom_call_target() ==
              kCubDeviceRadixSortUnassignedScratchSizeTarget ||
          custom_call->custom_call_target() ==
              kCubDeviceScanUnassignedScratchSizeTarget) {
        custom_calls.push_back(custom_call);
      }
    }
  }
  bool changed = false;
  for (auto* call : custom_calls) {
    if (call->custom_call_target() ==
        kCubDeviceRadixSortUnassignedScratchSizeTarget) {
      RETURN_IF_ERROR(RunOnSortInstruction(call));
      changed = true;
    } else if (call->custom_call_target() ==
               kCubDeviceScanUnassignedScratchSizeTarget) {
      RETURN_IF_ERROR(RunOnScanInstruction(call));
      changed = true;
    }
  }
  return changed;
}

// Replace compatible sort operations with custom calls.
absl::StatusOr<bool> EstimateCubScratchSize::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "EstimateCubScratchSize::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(
      3, "EstimateCubScratchSize::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
