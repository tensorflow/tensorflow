/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/base/macros.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

using MemoryDebugAnnotation ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::MemoryDebugAnnotation;  // NOLINT
using ScopedMemoryDebugAnnotation ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::ScopedMemoryDebugAnnotation;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
