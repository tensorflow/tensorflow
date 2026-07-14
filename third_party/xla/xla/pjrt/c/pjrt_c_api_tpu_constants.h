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

#ifndef XLA_PJRT_C_PJRT_C_API_TPU_CONSTANTS_H_
#define XLA_PJRT_C_PJRT_C_API_TPU_CONSTANTS_H_

namespace pjrt {

namespace tpu_configs {

constexpr char kMlFrameworkName[] = "ml_framework_name";
constexpr char kMlFrameworkVersion[] = "ml_framework_version";
constexpr char kMaxInflightComputations[] = "max_inflight_computations";
constexpr char kUseTfPjrtClient[] = "use_tf_pjrt_client";
constexpr char kUseGlobalTpuSystem[] = "use_global_tpu_system";
constexpr char kTpuAllowAsyncAllocations[] = "tpu_allow_async_allocations";
constexpr char kExecutableCompatibilityCheckOnDeserialization[] =
    "executable_compatibility_check_on_deserialization";
constexpr char kThrottleLowPriorityHostTransfers[] =
    "throttle_low_priority_host_transfers";
constexpr char kPinnedHostAllocationMode[] = "pinned_host_allocation_mode";
constexpr char kPremappedBufferSize[] = "premapped_buffer_size";
constexpr char kMaximumPremappedBufferSizeForTransfersInBytes[] =
    "maximum_premapped_buffer_size_for_transfers_in_bytes";
constexpr char kNumPremappedPartitions[] = "num_premapped_partitions";
constexpr char kSkipMegascalePjrtClient[] = "skip_megascale_pjrt_client";

}  // namespace tpu_configs

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_TPU_CONSTANTS_H_
