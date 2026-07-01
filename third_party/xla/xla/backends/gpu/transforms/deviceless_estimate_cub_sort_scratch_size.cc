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

#include "xla/backends/gpu/transforms/deviceless_estimate_cub_sort_scratch_size.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<int64_t>
DevicelessEstimateCubSortScratchSize::CalculateDevicelessScratchSize(
    HloCustomCallInstruction* custom_call, const Shape& key_shape,
    bool is_pairs, int64_t num_items, int64_t batch_size) {
  ASSIGN_OR_RETURN(const CubScratchSizeDevicelessLookup& registry,
                   CubScratchSizeDevicelessLookup::GetInstance());

  int32_t key_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(key_shape.element_type());
  std::optional<int32_t> value_type_size;
  if (is_pairs) {
    value_type_size = ShapeUtil::ByteSizeOfPrimitiveType(
        custom_call->operand(1)->shape().element_type());
  }

  std::optional<int64_t> lookup_result =
      registry.Lookup(cub_version_, device_name_, key_type_size,
                      value_type_size, num_items, batch_size);

  if (!lookup_result) {
    return absl::NotFoundError(absl::StrFormat(
        "No CUB scratch size entry found for version=%s, device=%s, "
        "key_size=%d, num_items=%d",
        cub_version_.ToString(), device_name_, key_type_size, num_items));
  }
  return *lookup_result;
}

absl::Status DevicelessEstimateCubSortScratchSize::RunOnSortInstruction(
    HloCustomCallInstruction* custom_call) {
  CHECK_EQ(custom_call->custom_call_target(),
           kCubDeviceRadixSortUnassignedScratchSizeTarget);

  const Shape& key_shape = custom_call->operand(0)->shape();
  bool is_pairs = custom_call->operand_count() == 2;
  int64_t num_items = Product(key_shape.dimensions());
  int64_t batch_size = num_items / key_shape.dimensions().back();
  ASSIGN_OR_RETURN(int64_t scratch_size, CalculateDevicelessScratchSize(
                                             custom_call, key_shape, is_pairs,
                                             num_items, batch_size));

  absl::string_view ffi_target =
      is_pairs ? kCubDeviceRadixSortPairsTarget : kCubDeviceRadixSortKeysTarget;
  ASSIGN_OR_RETURN(SortOptions sort_options,
                   custom_call->backend_config<SortOptions>());
  return CreateCubSortCustomCall(custom_call, scratch_size, ffi_target,
                                 sort_options.descending(), batch_size);
}

absl::StatusOr<bool> DevicelessEstimateCubSortScratchSize::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloCustomCallInstruction*> custom_calls;
  for (auto* inst : computation->instructions()) {
    if (auto custom_call = DynCast<HloCustomCallInstruction>(inst)) {
      if (custom_call->custom_call_target() ==
          kCubDeviceRadixSortUnassignedScratchSizeTarget) {
        custom_calls.push_back(custom_call);
      }
    }
  }
  bool changed = false;
  for (auto* call : custom_calls) {
    RETURN_IF_ERROR(RunOnSortInstruction(call));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> DevicelessEstimateCubSortScratchSize::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3,
                 "DevicelessEstimateCubSortScratchSize::RunImpl(), before:\n" +
                     module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(3,
                 "DevicelessEstimateCubSortScratchSize::RunImpl(), after:\n" +
                     module->ToString());
  return changed;
}

}  // namespace xla::gpu
