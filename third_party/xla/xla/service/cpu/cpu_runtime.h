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
extern const char* const kEigenMatMulF16SymbolName;
extern const char* const kEigenMatMulF32SymbolName;
extern const char* const kEigenMatMulF64SymbolName;
extern const char* const kEigenMatMulC64SymbolName;
extern const char* const kEigenMatMulC128SymbolName;
extern const char* const kEigenMatMulS32SymbolName;
extern const char* const kEigenMatMulU8SymbolName;
extern const char* const kEigenBatchMatMulF32SymbolName;
extern const char* const kACLConv2DF32SymbolName;
extern const char* const kACLMatMulF32SymbolName;
extern const char* const kACLBatchMatMulF32SymbolName;
extern const char* const kEigenConv2DF16SymbolName;
extern const char* const kEigenConv2DF32SymbolName;
extern const char* const kEigenConv3DF16SymbolName;
extern const char* const kEigenConv3DF32SymbolName;
extern const char* const kLegacyDuccFftSymbolName;
extern const char* const kDuccFftSymbolName;
extern const char* const kDuccSingleThreadedFftSymbolName;
extern const char* const kEigenSingleThreadedMatMulF16SymbolName;
extern const char* const kEigenSingleThreadedMatMulF32SymbolName;
extern const char* const kEigenSingleThreadedMatMulF64SymbolName;
extern const char* const kEigenSingleThreadedMatMulF8E4M3FNSymbolName;
extern const char* const kEigenSingleThreadedMatMulF8E5M2SymbolName;
extern const char* const kEigenSingleThreadedMatMulC64SymbolName;
extern const char* const kEigenSingleThreadedMatMulC128SymbolName;
extern const char* const kEigenSingleThreadedMatMulS32SymbolName;
extern const char* const kEigenSingleThreadedMatMulU8SymbolName;
extern const char* const kEigenSingleThreadedConv2DF16SymbolName;
extern const char* const kEigenSingleThreadedConv2DF32SymbolName;
extern const char* const kEigenSingleThreadedConv3DF16SymbolName;
extern const char* const kEigenSingleThreadedConv3DF32SymbolName;
extern const char* const kAcquireInfeedBufferForDequeueSymbolName;
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName;
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName;
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName;
extern const char* const kParallelForkJoinSymbolName;
extern const char* const kPrintfToStderrSymbolName;
extern const char* const kStatusIsSuccessSymbolName;
extern const char* const kKeyValueSortSymbolName;
extern const char* const kTopKF32SymbolName;
extern const char* const kAllReduceSymbolName;
extern const char* const kCollectivePermuteSymbolName;
extern const char* const kPartitionIdSymbolName;
extern const char* const kReplicaIdSymbolName;
extern const char* const kTracingStartSymbolName;
extern const char* const kTracingEndSymbolName;
extern const char* const kAllToAllSymbolName;
extern const char* const kAllGatherSymbolName;
extern const char* const kReduceScatterSymbolName;
extern const char* const kOneDnnMatMulSymbolName;
extern const char* const kOneDnnSoftmaxSymbolName;
extern const char* const kOneDnnLayerNormSymbolName;
extern const char* const kOneDnnConvolutionSymbolName;
extern const char* const kOneDnnMatMulReorderSymbolName;
extern const char* const kHandleFfiCallSymbolName;

// All symbol names for XLA CPU runtime functions need to start with this
// prefix.
extern const char* const kXlaCpuRuntimeSymbolNamePrefix;

int GetDeviceOrdinal(const xla::ExecutableRunOptions* run_options);

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_RUNTIME_H_
