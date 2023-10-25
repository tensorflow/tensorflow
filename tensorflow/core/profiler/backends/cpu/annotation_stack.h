/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_

#include <stddef.h>

#include <atomic>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::AnnotationStack;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
