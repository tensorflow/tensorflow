/* Copyright 2015 The OpenXLA Authors.

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

// Suite of types that represent device memory allocations. These are
// allocated by the StreamExecutor interface, which produces values appropriate
// for the underlying platform (whether it be CUDA or OpenCL).
//
// The untyped base class (like a device void*) is DeviceMemoryBase, which can
// be specialized for a given allocation type (like a device T*) using
// DeviceMemory<T>.

#ifndef XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_
#define XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_

#include <stddef.h>

#include "absl/base/macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/tensor_map.h"  // IWYU pragma: keep

namespace stream_executor {

using DeviceMemoryBase ABSL_DEPRECATE_AND_INLINE() =
    ::stream_executor::DeviceAddressBase;

template <typename T>
using DeviceMemory ABSL_DEPRECATE_AND_INLINE() =
    ::stream_executor::DeviceAddress<T>;

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_MEMORY_H_
