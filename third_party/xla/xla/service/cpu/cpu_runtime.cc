/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_runtime.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_cliques.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/collectives/in_process_collectives.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace cpu {
namespace runtime {

extern const char* const kEigenMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenMatMulF16";
extern const char* const kEigenMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenMatMulF32";
extern const char* const kEigenMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenMatMulF64";
extern const char* const kEigenMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenMatMulC64";
extern const char* const kEigenMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenMatMulC128";
extern const char* const kEigenMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenMatMulS32";
extern const char* const kEigenBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenBatchMatMulF32";
extern const char* const kACLConv2DF32SymbolName =
    "__xla_cpu_runtime_ACLConv2DF32";
extern const char* const kACLMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLMatMulF32";
extern const char* const kACLBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLBatchMatMulF32";
extern const char* const kEigenConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenConv2DF16";
extern const char* const kEigenConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenConv2DF32";
extern const char* const kEigenConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenConv3DF16";
extern const char* const kEigenConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenConv3DF32";
extern const char* const kEigenSingleThreadedMatMulF8E4M3FNSymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF8E4M3FN";
extern const char* const kEigenSingleThreadedMatMulF8E5M2SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF8E5M2";
extern const char* const kEigenSingleThreadedMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF16";
extern const char* const kEigenSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF32";
extern const char* const kEigenSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF64";
extern const char* const kEigenSingleThreadedMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC64";
extern const char* const kEigenSingleThreadedMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC128";
extern const char* const kEigenSingleThreadedMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulS32";
extern const char* const kEigenSingleThreadedMatMulU8SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulU8";
extern const char* const kEigenSingleThreadedConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF16";
extern const char* const kEigenSingleThreadedConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF32";
extern const char* const kEigenSingleThreadedConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF16";
extern const char* const kEigenSingleThreadedConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF32";
extern const char* const kAcquireInfeedBufferForDequeueSymbolName =
    "__xla_cpu_runtime_AcquireInfeedBufferForDequeue";
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName =
    "__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue";
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName =
    "__xla_cpu_runtime_AcquireOutfeedBufferForPopulation";
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName =
    "__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation";
extern const char* const kParallelForkJoinSymbolName =
    "__xla_cpu_runtime_ParallelForkJoin";
extern const char* const kPrintfToStderrSymbolName =
    "__xla_cpu_runtime_PrintfToStderr";
extern const char* const kStatusIsSuccessSymbolName =
    "__xla_cpu_runtime_StatusIsSuccess";
extern const char* const kKeyValueSortSymbolName =
    "__xla_cpu_runtime_KeyValueSort";
extern const char* const kTopKF32SymbolName = "__xla_cpu_runtime_TopKF32";
extern const char* const kTracingStartSymbolName =
    "__xla_cpu_runtime_TracingStart";
extern const char* const kTracingEndSymbolName = "__xla_cpu_runtime_TracingEnd";
extern const char* const kXlaCpuRuntimeSymbolNamePrefix = "__xla_cpu_runtime_";
extern const char* const kAllReduceSymbolName = "__xla_cpu_runtime_AllReduce";
extern const char* const kAllGatherSymbolName = "__xla_cpu_runtime_AllGather";
extern const char* const kReduceScatterSymbolName =
    "__xla_cpu_runtime_ReduceScatter";
extern const char* const kAllToAllSymbolName = "__xla_cpu_runtime_AllToAll";
extern const char* const kCollectivePermuteSymbolName =
    "__xla_cpu_runtime_CollectivePermute";
extern const char* const kPartitionIdSymbolName =
    "__xla_cpu_runtime_PartitionId";
extern const char* const kReplicaIdSymbolName = "__xla_cpu_runtime_ReplicaId";
extern const char* const kHandleFfiCallSymbolName =
    "__xla_cpu_runtime_HandleFfiCall";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
