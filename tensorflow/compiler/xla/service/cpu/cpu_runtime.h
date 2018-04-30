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

#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
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
extern const char* const kMKLConvF32SymbolName;
extern const char* const kMKLMatMulF32SymbolName;
extern const char* const kMKLMatMulF64SymbolName;
extern const char* const kMKLSingleThreadedMatMulF32SymbolName;
extern const char* const kMKLSingleThreadedMatMulF64SymbolName;
extern const char* const kEigenConvF16SymbolName;
extern const char* const kEigenConvF32SymbolName;
extern const char* const kEigenFftSymbolName;
extern const char* const kEigenSingleThreadedMatMulF16SymbolName;
extern const char* const kEigenSingleThreadedMatMulF32SymbolName;
extern const char* const kEigenSingleThreadedMatMulF64SymbolName;
extern const char* const kEigenSingleThreadedConvF16SymbolName;
extern const char* const kEigenSingleThreadedConvF32SymbolName;
extern const char* const kAcquireInfeedBufferForDequeueSymbolName;
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName;
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName;
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName;
extern const char* const kParallelForkJoinSymbolName;

// All symbol names for XLA CPU runtime functions need to start with this
// prefix.
extern const char* const kXlaCpuRuntimeSymbolNamePrefix;

// Returns the infeed manager used by the CPU runtime.
XfeedManager* GetXfeedManager();

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

extern "C" {

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
    xla::int32 buffer_length, const void* shape, xla::int32 shape_length);

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
    xla::int32 buffer_length, void* buffer_ptr, const void* shape_ptr,
    xla::int32 shape_length);

// Blocks until the next outfeed buffer is available to be populated, then
// returns it.
extern void* __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
    xla::int32 buffer_length, const void* shape_ptr, xla::int32 shape_length);

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
    xla::int32 buffer_length, void* buffer_ptr, const void* shape_ptr,
    xla::int32 shape_length);

}  // extern "C"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_H_
