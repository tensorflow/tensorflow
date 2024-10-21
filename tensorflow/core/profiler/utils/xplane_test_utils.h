/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_TEST_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_TEST_UTILS_H_

#include <initializer_list>

#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::CreateTfFunctionCallEvent;  // NOLINT
using tsl::profiler::CreateXEvent;               // NOLINT
using tsl::profiler::GetOrCreateGpuXPlane;       // NOLINT
using tsl::profiler::GetOrCreateHostXPlane;      // NOLINT
using tsl::profiler::GetOrCreateTpuXPlane;       // NOLINT
using tsl::profiler::XStatValue;                 // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_TEST_UTILS_H_
