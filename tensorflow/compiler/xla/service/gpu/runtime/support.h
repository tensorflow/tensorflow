/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

// Disable all CustomCall checks in optimized build.
inline constexpr runtime::CustomCall::RuntimeChecks checks =  // NOLINT
#if defined(NDEBUG)
    runtime::CustomCall::RuntimeChecks::kNone;
#else
    runtime::CustomCall::RuntimeChecks::kDefault;
#endif

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::FlatMemrefView& memref) {
  return se::DeviceMemoryBase(memref.data, memref.size_in_bytes);
}

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::MemrefView& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::StridedMemrefView& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_
