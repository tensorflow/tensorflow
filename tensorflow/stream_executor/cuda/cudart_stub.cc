/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// This file wraps cuda runtime calls with dso loader so that we don't need to
// have explicit linking to libcuda.

#include "cuda/include/cuda_runtime_api.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

namespace {
void *GetDsoHandle() {
  static auto handle = [] {
    void *result = nullptr;
    using DsoLoader = stream_executor::internal::DsoLoader;
    DsoLoader::GetLibcudartDsoHandle(&result).IgnoreError();
    return result;
  }();
  return handle;
}

template <typename T>
T LoadSymbol(const char *symbol_name) {
  void *symbol = nullptr;
  auto env = stream_executor::port::Env::Default();
  env->GetSymbolFromLibrary(GetDsoHandle(), symbol_name, &symbol).IgnoreError();
  return reinterpret_cast<T>(symbol);
}
cudaError_t GetSymbolNotFoundError() {
  return cudaErrorSharedObjectSymbolNotFound;
}
const char *GetSymbolNotFoundStrError() {
  return "cudaErrorSharedObjectSymbolNotFound";
}
}  // namespace

// Code below is auto-generated.
extern "C" {
cudaError_t CUDART_CB cudaFree(void *devPtr) {
  using FuncPtr = cudaError_t (*)(void *devPtr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaFree");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(devPtr);
}

cudaError_t CUDART_CB cudaGetDevice(int *device) {
  using FuncPtr = cudaError_t (*)(int *device);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaGetDevice");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device);
}

cudaError_t CUDART_CB cudaGetDeviceProperties(cudaDeviceProp *prop,
                                              int device) {
  using FuncPtr = cudaError_t (*)(cudaDeviceProp * prop, int device);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaGetDeviceProperties");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(prop, device);
}

const char *CUDART_CB cudaGetErrorString(cudaError_t error) {
  using FuncPtr = const char *(*)(cudaError_t error);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaGetErrorString");
  if (!func_ptr) return GetSymbolNotFoundStrError();
  return func_ptr(error);
}

cudaError_t CUDART_CB cudaSetDevice(int device) {
  using FuncPtr = cudaError_t (*)(int device);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaSetDevice");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(device);
}

cudaError_t CUDART_CB cudaStreamAddCallback(cudaStream_t stream,
                                            cudaStreamCallback_t callback,
                                            void *userData,
                                            unsigned int flags) {
  using FuncPtr =
      cudaError_t (*)(cudaStream_t stream, cudaStreamCallback_t callback,
                      void *userData, unsigned int flags);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaStreamAddCallback");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(stream, callback, userData, flags);
}

cudaError_t CUDART_CB cudaGetDeviceCount(int *count) {
  using FuncPtr = cudaError_t (*)(int *count);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaGetDeviceCount");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(count);
}

cudaError_t CUDART_CB cudaPointerGetAttributes(
    struct cudaPointerAttributes *attributes, const void *ptr) {
  using FuncPtr = cudaError_t (*)(struct cudaPointerAttributes * attributes,
                                  const void *ptr);
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaPointerGetAttributes");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(attributes, ptr);
}

cudaError_t CUDART_CB cudaGetLastError() {
  using FuncPtr = cudaError_t (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cudaGetLastError");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}
}  // extern "C"
