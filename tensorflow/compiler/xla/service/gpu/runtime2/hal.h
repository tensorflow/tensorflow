/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_HAL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_HAL_H_

#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Helper functions to work with IREE buffers and buffer views
//===----------------------------------------------------------------------===//

Shape GetBufferShape(iree_hal_buffer_view_t* view);

StatusOr<se::DeviceMemoryBase> GetDeviceMemory(
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_t* buffer);
StatusOr<se::DeviceMemoryBase> GetDeviceMemory(
    iree_hal_allocator_t* device_allocator, iree_hal_buffer_view_t* view);

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_HAL_H_
