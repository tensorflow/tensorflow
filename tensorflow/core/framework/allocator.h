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

#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_

#include <stdlib.h>

#include <functional>
#include <limits>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/framework/allocator.h"

namespace tensorflow {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::AllocationAttributes;
using tsl::Allocator;
using tsl::AllocatorAttributes;
using tsl::AllocatorMemoryType;
using tsl::AllocatorStats;
using tsl::AllocatorWrapper;
using tsl::cpu_allocator;
using tsl::cpu_allocator_base;
using tsl::CPUAllocatorFullStatsEnabled;
using tsl::CPUAllocatorStatsEnabled;
using tsl::DisableCPUAllocatorStats;
using tsl::EnableCPUAllocatorFullStats;
using tsl::EnableCPUAllocatorStats;
using tsl::SubAllocator;
// NOLINTEND(misc-unused-using-decls)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
