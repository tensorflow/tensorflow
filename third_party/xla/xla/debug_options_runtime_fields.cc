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

#include "xla/debug_options_runtime_fields.h"

#include <cstddef>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

// Sorted by field number; the number is in the comment.
constexpr int kRuntimeFields[] = {
    DebugOptions::kXlaForceHostPlatformDeviceCountFieldNumber,            // 102
    DebugOptions::kXlaHloEvaluatorUseFastPathFieldNumber,                 // 106
    DebugOptions::kXlaDumpToFieldNumber,                                  // 109
    DebugOptions::kXlaGpuNcclTerminationTimeoutSecondsFieldNumber,        // 163
    DebugOptions::kXlaGpuGraphMinGraphSizeFieldNumber,                    // 208
    DebugOptions::kXlaGpuEnableHighestPriorityAsyncStreamFieldNumber,     // 216
    DebugOptions::kXlaGpuEnableCommandBufferFieldNumber,                  // 258
    DebugOptions::kXlaGpuEnableNcclCommSplittingFieldNumber,              // 272
    DebugOptions::kXlaGpuNcclCollectiveMaxNchannelsFieldNumber,           // 273
    DebugOptions::kXlaGpuNcclP2PMaxNchannelsFieldNumber,                  // 274
    DebugOptions::kXlaGpuNcclInitMaxRankPerRootRatioFieldNumber,          // 277
    DebugOptions::kXlaGpuNcclTerminateOnErrorFieldNumber,                 // 301
    DebugOptions::kXlaCmdBufferTraceCacheSizeFieldNumber,                 // 311
    DebugOptions::kXlaGpuTempBufferUseSeparateColorFieldNumber,           // 312
    DebugOptions::kXlaEnableCommandBuffersDuringProfilingFieldNumber,     // 317
    DebugOptions::kXlaGpuExecutableWarnStuckTimeoutSecondsFieldNumber,    // 327
    DebugOptions::kXlaGpuExecutableTerminateTimeoutSecondsFieldNumber,    // 328
    DebugOptions::                                                        //
    kXlaGpuExperimentalParallelCollectiveOverlapLimitFieldNumber,         // 336
    DebugOptions::kXlaGpuRequireExclusiveLockFieldNumber,                 // 347
    DebugOptions::kXlaGpuNcclBlockingCommunicatorsFieldNumber,            // 390
    DebugOptions::                                                        //
    kXlaGpuFirstCollectiveCallWarnStuckTimeoutSecondsFieldNumber,         // 391
    DebugOptions::                                                        //
    kXlaGpuFirstCollectiveCallTerminateTimeoutSecondsFieldNumber,         // 392
    DebugOptions::kXlaGpuNcclAsyncExecutionFieldNumber,                   // 393
    DebugOptions::kXlaEarlyExitWithLayoutsFieldNumber,                    // 397
    DebugOptions::kXlaGpuCommandBufferSchedulingModeFieldNumber,          // 404
    DebugOptions::                                                        //
    kXlaGpuExperimentalEnableNcclSymmetricBuffersFieldNumber,             // 406
    DebugOptions::kXlaGpuCommandBufferUnrollLoopsFieldNumber,             // 411
    DebugOptions::                                                        //
    kXlaGpuExperimentalEnableChecksumTracingOnThunksFieldNumber,          // 414
    DebugOptions::                                                        //
    kXlaCpuCollectiveCallTerminateTimeoutSecondsFieldNumber,              // 417
    DebugOptions::kXlaCpuCollectiveCallWarnStuckSecondsFieldNumber,       // 418
    DebugOptions::kXlaGpuExperimentalThunkBufferDebugFilterFieldNumber,   // 424
    DebugOptions::kXlaGpuDetectNanFieldNumber,                            // 426
    DebugOptions::kXlaGpuDetectInfFieldNumber,                            // 428
    DebugOptions::                                                        //
    kXlaGpuExperimentalEnableBufferSaverOnThunksFieldNumber,              // 431
    DebugOptions::kXlaGpuExecutableEmbedDebugInfoFieldNumber,             // 437
    DebugOptions::kXlaCpuCollectiveTimeoutSecondsFieldNumber,             // 438
    DebugOptions::kXlaGpuExecutionTerminateTimeoutFieldNumber,            // 451
    DebugOptions::kXlaGpuLogMinmaxFieldNumber,                            // 453
    DebugOptions::kXlaGpuExecutionProgressTrackingFieldNumber,            // 457
    DebugOptions::kXlaGpuCollectivesImplementationFieldNumber,            // 468
    DebugOptions::kXlaGpuCommandBufferUpdateModeFieldNumber,              // 469
    DebugOptions::kXlaGpuExecutableNumComputeStreamsFieldNumber,          // 486
    DebugOptions::kXlaGpuExecutableNumCommunicationStreamsFieldNumber,    // 487
    DebugOptions::                                                        //
    kXlaGpuExperimentalThunkBufferDebugModuleOutputsFieldNumber,          // 492
    DebugOptions::kXlaGpuEnableAllocatorSpatialPartitioningFieldNumber,   // 494
    DebugOptions::                                                        //
    kXlaGpuEnableCollectivesCommandBufferFilterFieldNumber,               // 497
    DebugOptions::                                                        //
    kXlaEnableNcclSymmetricBuffersForCollectivesFieldNumber,              // 503
    DebugOptions::kXlaGpuEnableNcclUserBuffersInDefaultSpaceFieldNumber,  // 506
    DebugOptions::kXlaGpuGxlScratchSizeBytesFieldNumber,                  // 514
    DebugOptions::kXlaGpuExperimentalEarlyExitFieldNumber,                // 549
    DebugOptions::kXlaGpuExperimentalVmmDisabledFieldNumber,              // 551
    DebugOptions::kXlaGpuDeviceExecutionTerminateTimeoutFieldNumber,      // 553
    DebugOptions::kXlaGpuMemoryFractionPolicyFieldNumber,                 // 560
    DebugOptions::kXlaGpuExperimentalNcclWindowSymmetricFieldNumber,      // 561
};

// Hand-written because std::is_sorted and std::adjacent_find are not
// constexpr in C++17, which the OSS build uses.
constexpr bool IsStrictlyIncreasing(absl::Span<const int> fields) {
  for (size_t i = 1; i < fields.size(); ++i) {
    if (fields[i - 1] >= fields[i]) {
      return false;
    }
  }
  return true;
}

static_assert(IsStrictlyIncreasing(kRuntimeFields),
              "kRuntimeFields must be sorted by field number, without "
              "duplicates.");

}  // namespace

absl::Span<const int> RuntimeDebugOptionsFieldNumbers() {
  return kRuntimeFields;
}

bool IsRuntimeDebugOptionsField(int field_number) {
  return absl::c_binary_search(kRuntimeFields, field_number);
}

}  // namespace xla
