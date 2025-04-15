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

#ifndef TENSORFLOW_CORE_PLATFORM_MEM_H_
#define TENSORFLOW_CORE_PLATFORM_MEM_H_

#include "tsl/platform/mem.h"
// TODO(cwhipkey): remove this when callers use annotations directly.
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {
namespace port {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::port::AlignedFree;
using ::tsl::port::AlignedMalloc;
using ::tsl::port::AvailableRam;
using ::tsl::port::Free;
using ::tsl::port::GetMemoryBandwidthInfo;
using ::tsl::port::GetMemoryInfo;
using ::tsl::port::Malloc;
using ::tsl::port::MallocExtension_GetAllocatedSize;
using ::tsl::port::MallocExtension_ReleaseToSystem;
using ::tsl::port::MemoryBandwidthInfo;
using ::tsl::port::MemoryInfo;
using ::tsl::port::Realloc;
// NOLINTEND(misc-unused-using-decls)
}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_MEM_H_
