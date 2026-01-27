/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_
#define XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_

#include "absl/base/macros.h"
#include "xla/stream_executor/device_address_handle.h"

namespace stream_executor {

using DeviceMemoryHandle ABSL_DEPRECATE_AND_INLINE() =
    ::stream_executor::DeviceAddressHandle;

}

#endif  // XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_
