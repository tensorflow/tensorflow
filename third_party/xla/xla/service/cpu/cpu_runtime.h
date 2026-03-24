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

// This header declares functions which may be called by the generated code on
// the CPU. Calls to these functions must be resolved explicitly in the JIT in
// xla::cpu::SimpleResolver.  It also defines a per-CpuExecutable context
// which is used to cache expensive state and resources utilized by the
// aforementioned functions.
//
// Other functions are declared in individual libraries as well, such as
// runtime_conv2d and runtime_matmul. As individual libraries, callers for
// ahead-of-time compilation can link only the required subset.

#ifndef XLA_SERVICE_CPU_CPU_RUNTIME_H_
#define XLA_SERVICE_CPU_CPU_RUNTIME_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/executable_run_options.h"

namespace xla {
namespace cpu {
namespace runtime {

// Names of runtime functions. These get resolved from the generated code to the
// right symbol at link time in one of two ways:
// 1. When using the JIT, the symbol resolver (xla::cpu::RuntimeSymbolGenerator)
//    maps this symbol name to the actual symbol.
// 2. When using ahead-of-time compilation, the linker can resolve the name
//    because it is a symbol in the cpu_runtime library.
inline constexpr absl::string_view kEigenMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenMatMulF16";
inline constexpr absl::string_view kEigenMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenMatMulF32";
inline constexpr absl::string_view kEigenMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenMatMulF64";
inline constexpr absl::string_view kEigenMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenMatMulC64";
inline constexpr absl::string_view kEigenMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenMatMulC128";
inline constexpr absl::string_view kEigenMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenMatMulS32";
extern const char* const kEigenMatMulU8SymbolName;
inline constexpr absl::string_view kEigenBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenBatchMatMulF32";
inline constexpr absl::string_view kACLConv2DF32SymbolName =
    "__xla_cpu_runtime_ACLConv2DF32";
inline constexpr absl::string_view kACLMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLMatMulF32";
inline constexpr absl::string_view kACLBatchMatMulF32SymbolName =
    "__xla_cpu_runtime_ACLBatchMatMulF32";
inline constexpr absl::string_view kEigenConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenConv2DF16";
inline constexpr absl::string_view kEigenConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenConv2DF32";
inline constexpr absl::string_view kEigenConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenConv3DF16";
inline constexpr absl::string_view kEigenConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenConv3DF32";
inline constexpr absl::string_view kEigenSingleThreadedMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF16";
inline constexpr absl::string_view kEigenSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF32";
inline constexpr absl::string_view kEigenSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF64";
inline constexpr absl::string_view
    kEigenSingleThreadedMatMulF8E4M3FNSymbolName =
        "__xla_cpu_runtime_EigenSingleThreadedMatMulF8E4M3FN";
inline constexpr absl::string_view kEigenSingleThreadedMatMulF8E5M2SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF8E5M2";
inline constexpr absl::string_view kEigenSingleThreadedMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC64";
inline constexpr absl::string_view kEigenSingleThreadedMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC128";
inline constexpr absl::string_view kEigenSingleThreadedMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulS32";
inline constexpr absl::string_view kEigenSingleThreadedMatMulU8SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulU8";
inline constexpr absl::string_view kEigenSingleThreadedConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF16";
inline constexpr absl::string_view kEigenSingleThreadedConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF32";
inline constexpr absl::string_view kEigenSingleThreadedConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF16";
inline constexpr absl::string_view kEigenSingleThreadedConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF32";
inline constexpr absl::string_view kAcquireInfeedBufferForDequeueSymbolName =
    "__xla_cpu_runtime_AcquireInfeedBufferForDequeue";
inline constexpr absl::string_view kReleaseInfeedBufferAfterDequeueSymbolName =
    "__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue";
inline constexpr absl::string_view
    kAcquireOutfeedBufferForPopulationSymbolName =
        "__xla_cpu_runtime_AcquireOutfeedBufferForPopulation";
inline constexpr absl::string_view
    kReleaseOutfeedBufferAfterPopulationSymbolName =
        "__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation";
inline constexpr absl::string_view kParallelForkJoinSymbolName =
    "__xla_cpu_runtime_ParallelForkJoin";
inline constexpr absl::string_view kPrintfToStderrSymbolName =
    "__xla_cpu_runtime_PrintfToStderr";
inline constexpr absl::string_view kStatusIsSuccessSymbolName =
    "__xla_cpu_runtime_StatusIsSuccess";
inline constexpr absl::string_view kKeyValueSortSymbolName =
    "__xla_cpu_runtime_KeyValueSort";
inline constexpr absl::string_view kTopKF32SymbolName =
    "__xla_cpu_runtime_TopKF32";
inline constexpr absl::string_view kAllReduceSymbolName =
    "__xla_cpu_runtime_AllReduce";
inline constexpr absl::string_view kCollectivePermuteSymbolName =
    "__xla_cpu_runtime_CollectivePermute";
inline constexpr absl::string_view kPartitionIdSymbolName =
    "__xla_cpu_runtime_PartitionId";
inline constexpr absl::string_view kReplicaIdSymbolName =
    "__xla_cpu_runtime_ReplicaId";
inline constexpr absl::string_view kTracingStartSymbolName =
    "__xla_cpu_runtime_TracingStart";
inline constexpr absl::string_view kTracingEndSymbolName =
    "__xla_cpu_runtime_TracingEnd";
inline constexpr absl::string_view kAllToAllSymbolName =
    "__xla_cpu_runtime_AllToAll";
inline constexpr absl::string_view kAllGatherSymbolName =
    "__xla_cpu_runtime_AllGather";
inline constexpr absl::string_view kReduceScatterSymbolName =
    "__xla_cpu_runtime_ReduceScatter";
inline constexpr absl::string_view kHandleFfiCallSymbolName =
    "__xla_cpu_runtime_HandleFfiCall";

// All symbol names for XLA CPU runtime functions need to start with this
// prefix.
inline constexpr absl::string_view kXlaCpuRuntimeSymbolNamePrefix =
    "__xla_cpu_runtime_";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_RUNTIME_H_
