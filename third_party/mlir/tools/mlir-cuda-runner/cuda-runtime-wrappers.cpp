//===- cuda-runtime-wrappers.cpp - MLIR CUDA runner wrapper library -------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <memory.h>

#include "llvm/Support/raw_ostream.h"

#include "cuda.h"

namespace {
int32_t reportErrorIfAny(CUresult result, const char *where) {
  if (result != CUDA_SUCCESS) {
    llvm::errs() << "CUDA failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mcuModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      cuModuleLoadData(reinterpret_cast<CUmodule *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mcuModuleGetFunction(void **function, void *module,
                                        const char *name) {
  return reportErrorIfAny(
      cuModuleGetFunction(reinterpret_cast<CUfunction *>(function),
                          reinterpret_cast<CUmodule>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mcuLaunchKernel(void *function, intptr_t gridX,
                                   intptr_t gridY, intptr_t gridZ,
                                   intptr_t blockX, intptr_t blockY,
                                   intptr_t blockZ, int32_t smem, void *stream,
                                   void **params, void **extra) {
  return reportErrorIfAny(
      cuLaunchKernel(reinterpret_cast<CUfunction>(function), gridX, gridY,
                     gridZ, blockX, blockY, blockZ, smem,
                     reinterpret_cast<CUstream>(stream), params, extra),
      "LaunchKernel");
}

extern "C" void *mcuGetStreamHelper() {
  CUstream stream;
  reportErrorIfAny(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "StreamCreate");
  return stream;
}

extern "C" int32_t mcuStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "StreamSync");
}

/// Helper functions for writing mlir example code

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N> struct MemRefType {
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// Allows to register a pointer with the CUDA runtime. Helpful until
// we have transfer functions implemented.
extern "C" void mcuMemHostRegister(const MemRefType<float, 1> *arg,
                                   int32_t flags) {
  reportErrorIfAny(
      cuMemHostRegister(arg->data, arg->sizes[0] * sizeof(float), flags),
      "MemHostRegister");
  for (int pos = 0; pos < arg->sizes[0]; pos++) {
    arg->data[pos] = 1.23f;
  }
}

// Allows to register a pointer with the CUDA runtime. Helpful until
// we have transfer functions implemented.
extern "C" void mcuMemHostRegisterPtr(void *ptr, int32_t flags) {
  reportErrorIfAny(cuMemHostRegister(ptr, sizeof(void *), flags),
                   "MemHostRegister");
}

/// Prints the given float array to stderr.
extern "C" void mcuPrintFloat(const MemRefType<float, 1> *arg) {
  if (arg->sizes[0] == 0) {
    llvm::outs() << "[]\n";
    return;
  }
  llvm::outs() << "[" << arg->data[0];
  for (int pos = 1; pos < arg->sizes[0]; pos++) {
    llvm::outs() << ", " << arg->data[pos];
  }
  llvm::outs() << "]\n";
}
