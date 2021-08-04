/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_H_

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {
namespace runtime {

// Names of runtime functions. These get resolved from the generated code to the
// right symbol at link time in one of two ways:
// 1. When using the JIT, the symbol resolver (SimpleResolver in
//    third_party/tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc) maps
//    this symbol name to
//    the actual symbol.
// 2. When using ahead-of-time compilation, the linker can resolve the name
//    because it is a symbol in the cpu_runtime library.
extern const char* const kEigenMatMulF16SymbolName;
extern const char* const kEigenMatMulF32SymbolName;
extern const char* const kEigenMatMulF64SymbolName;
extern const char* const kEigenMatMulC64SymbolName;
extern const char* const kEigenMatMulC128SymbolName;
extern const char* const kEigenMatMulS32SymbolName;
extern const char* const kMKLConvF32SymbolName;
extern const char* const kMKLMatMulF32SymbolName;
extern const char* const kMKLMatMulF64SymbolName;
extern const char* const kMKLSingleThreadedMatMulF32SymbolName;
extern const char* const kMKLSingleThreadedMatMulF64SymbolName;
extern const char* const kEigenConvF16SymbolName;
extern const char* const kEigenConvF32SymbolName;
extern const char* const kEigenFftSymbolName;
extern const char* const kEigenSingleThreadedFftSymbolName;
extern const char* const kEigenSingleThreadedMatMulF16SymbolName;
extern const char* const kEigenSingleThreadedMatMulF32SymbolName;
extern const char* const kEigenSingleThreadedMatMulF64SymbolName;
extern const char* const kEigenSingleThreadedMatMulC64SymbolName;
extern const char* const kEigenSingleThreadedMatMulC128SymbolName;
extern const char* const kEigenSingleThreadedMatMulS32SymbolName;
extern const char* const kEigenSingleThreadedConvF16SymbolName;
extern const char* const kEigenSingleThreadedConvF32SymbolName;
extern const char* const kAcquireInfeedBufferForDequeueSymbolName;
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName;
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName;
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName;
extern const char* const kParallelForkJoinSymbolName;
extern const char* const kPrintfToStderrSymbolName;
extern const char* const kKeyValueSortSymbolName;
extern const char* const kTopKF32SymbolName;
extern const char* const kAllReduceSymbolName;
extern const char* const kCollectivePermuteSymbolName;
extern const char* const kReplicaIdSymbolName;
extern const char* const kTracingStartSymbolName;
extern const char* const kTracingEndSymbolName;
extern const char* const kAllToAllSymbolName;

// All symbol names for XLA CPU runtime functions need to start with this
// prefix.
extern const char* const kXlaCpuRuntimeSymbolNamePrefix;

// Returns the infeed manager used by the CPU runtime for the CPU device
// `device_ordinal`.  Note the device ordinal does not name a CPU
XfeedManager* GetXfeedManager(int device_ordinal);

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

extern "C" {

extern int __xla_cpu_runtime_PrintfToStderr(const char* format, ...);

extern xla::int64 __xla_cpu_runtime_TracingStart(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    const char* name);
extern void __xla_cpu_runtime_TracingEnd(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    xla::int64 id);

// Some things common to all of the runtime entry points below:
//
//  * The shape pointer and shape_length reflect values that can be deserialized
//    via llvm_ir::DecodeSelfDescribingShapeConstant. This is the way we pass
//    reified type information from the generated program to the runtime, which
//    helps check the type safety and contract for the emitted-code/runtime
//    communication.
//
//  * run_options is used to look up the device ordinal for the stream executor
//    we're executing under.  If it is null the device ordinal is assumed to be
//    0 (this behavior helps in writing tests).

// Note: in the runtime entry points below, the shape pointer and shape_length
// reflect values that can be deserialized via
// llvm_ir::DecodeSelfDescribingShapeConstant. This is the way we pass reified
// type information from the generated program to the runtime, which helps check
// the type safety and contract for the emitted-code/runtime communication.

// Blocks until the next infeed buffer is ready to be dequeued, then
// returns it. Fails catastrophically if the next enqueued buffer is
// not of the correct length in bytes. Checking the shape rather than
// the length would be more exact, but the length check is chosen as a
// tradeoff between error checking and speed/simplicity.
extern void* __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    const void* shape, xla::int32 shape_length);

// Relinquishes the next infeed buffer that was returned by
// __xla_cpu_runtime_AcquireInfeedBufferForDequeue. Once this call
// completes the data at buffer_ptr may no longer be
// accessed. buffer_length must match the length passed to the call to
// __xla_cpu_runtime_AcquireInfeedBufferForDequeue that returned
// buffer_ptr. This function must be called before the next buffer is
// acquired, i.e., there may only be one outstanding infeed buffer in
// use by the runtime.  TODO(b/31340454) investigate whether or not it
// is worth supporting zero-copy infeed where the buffer is retained
// by the compiled code until it has been used. If zero-copy infeed is
// implemented we will add support for multiple outstanding buffers
// that can be returned out of order.
extern void __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    void* buffer_ptr, const void* shape_ptr, xla::int32 shape_length);

// Blocks until the next outfeed buffer is available to be populated, then
// returns it.
extern void* __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    const void* shape_ptr, xla::int32 shape_length);

// Relinquishes the outfeed buffer after it has been populated.
// buffer_ptr must have been previously returned by
// __xla_cpu_runtime_AcquireOutfeedBufferForPopulation.
// Once this call completes, buffer_ptr may no longer be accessed.
// buffer_length must match the length passed to the call to
// __xla_cpu_runtime_AcquireInfeedBufferForDequeue that returned
// buffer_ptr. This function must be called before the next buffer is
// acquired, i.e., there may only be one outstanding outfeed buffer in
// use by the runtime.
extern void __xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    void* buffer_ptr, const void* shape_ptr, xla::int32 shape_length);

// Perform all reduce on a CPU.
//
// participating_replicas: array of replica IDs participating in the reduction,
// cf. GetParticipatingIDs.
// channel_id_present, op_id: whether op_id is a channel ID or a module ID.
// reduction_kind: operator used for a reduction, cf. ReductionKind.
// shape_ptr: shape of all input/output buffers.
extern void __xla_cpu_runtime_AllReduce(
    const xla::ExecutableRunOptions* run_options,
    const void* replica_groups_str, xla::int32 replica_groups_str_size,
    xla::int32 channel_id_present, xla::int64 op_id, xla::int32 reduction_kind,
    const void* shape_ptr, xla::int32 shape_length, xla::int32 num_buffers,
    void** input_buffers, void** output_buffers);

extern void __xla_cpu_runtime_CollectivePermute(
    const xla::ExecutableRunOptions* run_options, xla::int32 channel_id_present,
    xla::int64 op_id, xla::int32 byte_size, void* input_buffer,
    void* output_buffer, const void* source_target_pairs,
    xla::int32 source_target_pairs_size);

extern void __xla_cpu_runtime_AllToAll(
    const xla::ExecutableRunOptions* run_options, xla::int32 channel_id_present,
    xla::int64 op_id, const void* replica_groups_str,
    xla::int32 replica_groups_str_size, xla::int32 num_buffers,
    xla::int64 buffer_size, void** source_buffers, void** destination_buffers);

// Write the replica ID into the output buffer.
extern void __xla_cpu_runtime_ReplicaId(
    const xla::ExecutableRunOptions* run_options, void* output_buffer);

}  // extern "C"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_H_
