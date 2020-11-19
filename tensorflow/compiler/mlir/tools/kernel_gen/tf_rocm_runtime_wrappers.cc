/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Implements C wrappers around the ROCm library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.

#include <cassert>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"

#define HIP_REPORT_IF_ERROR(expr)                                       \
  [](hipError_t result) {                                               \
    if (!result) return;                                                \
    const char* name = hipGetErrorName(result);                         \
    if (!name) name = "<unknown>";                                      \
    llvm::errs() << "'" << #expr << "' failed with '" << name << "'\n"; \
  }(expr)

extern "C" hipModule_t mgpuModuleLoad(void* data) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char* name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void** params,
                                 void** extra) {
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
}

extern "C" hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void* ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegisterMemRef(
    int64_t rank, StridedMemRefType<char, 1>* descriptor,
    int64_t elementSizeBytes) {
  llvm::SmallVector<int64_t, 4> denseStrides(rank);
  llvm::ArrayRef<int64_t> sizes(descriptor->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

#endif  // TENSORFLOW_USE_ROCM
