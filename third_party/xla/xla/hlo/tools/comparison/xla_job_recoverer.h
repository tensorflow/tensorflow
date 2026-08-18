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

#ifndef XLA_HLO_TOOLS_COMPARISON_XLA_JOB_RECOVERER_H_
#define XLA_HLO_TOOLS_COMPARISON_XLA_JOB_RECOVERER_H_

#include <functional>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_calculator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_propagator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"

namespace xla::numerics::comparison {

struct XlaJobRecovererData;

// Recovers original tensor summaries from runtime tensor logs of an XLA job.
//
// This class processes tensor summaries logged during the execution of an
// optimized HLO module, reverses the transformations performed by XLA
// optimizations, and recovers summaries corresponding to tensors in the
// *original* HLO module. It leverages `OriginalTensorSummaryCalculator` to
// perform the initial recovery, `OriginalTensorSummarySequencer` to reorder
// summaries based on HLO data dependencies, and
// `OriginalTensorSummaryPropagator` to propagate summaries across
// value-preserving instructions (e.g., reshape, transpose), filling in gaps
// where logging might be sparse.
//
// For each replica in the job, it calls the user-provided callback with
// sequenced and propagated tensor summaries.
class XlaJobRecoverer {
 public:
  using OriginalTensorSummaryCallbackGetter =
      std::function<OriginalTensorSummaryCallback(int replica_id)>;

  // Represents a tensor summary from a single device shard before device
  // assignment is applied.
  struct DeviceTensorSummary {
    // The logical device ID. This is before the DeviceAssignment is applied.
    GlobalDeviceId logical_device_id;
    ::xla::comparison::FloatSummary summary;
  };

  // Creates an XlaJobRecoverer instance.
  //
  // Parameters:
  //   device_assignment: The device assignment used in the XLA job.
  //   original_module: The HLO module before optimization.
  //   optimized_module: The HLO module after optimization, which was executed
  //     to produce the tensor summaries.
  //   callback_getter: A function that returns a callback to be invoked for
  //     each recovered tensor summary for a given replica_id.
  //   temp_file_base_path: A path prefix for temporary files used during
  //     sequencing and propagation.
  //   comparison_variant: The variant of the comparison (baseline or target).
  static absl::StatusOr<
      std::pair<std::unique_ptr<XlaJobRecoverer>,
                OriginalTensorSummaryCalculator::CreationMetrics>>
  Create(std::unique_ptr<const xla::DeviceAssignment> device_assignment,
         HloModule* original_module, HloModule* optimized_module,
         OriginalTensorSummaryCallbackGetter&& callback_getter,
         absl::string_view temp_file_base_path,
         absl::string_view sequenced_file_base_path,
         ComparisonVariant comparison_variant);

  // Processes a tensor summary from a single shard of a device.
  absl::Status ProcessDeviceTensorSummary(
      const AbsoluteScopedTensorKey& optimized_tensor_position,
      DeviceTensorSummary shard_summary);

  // Finishes recovery process. This triggers sequencing of raw summaries and
  // propagation, during which `callback` passed to `Create` will be invoked.
  absl::StatusOr<
      std::vector<OriginalTensorSummaryPropagator::ProcessingMetrics>>
  Finish();

  ~XlaJobRecoverer();

 private:
  explicit XlaJobRecoverer(std::unique_ptr<XlaJobRecovererData> data);
  std::unique_ptr<XlaJobRecovererData> data_;
};

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_XLA_JOB_RECOVERER_H_
