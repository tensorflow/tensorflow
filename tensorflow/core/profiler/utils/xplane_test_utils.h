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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

using XStatValue = absl::variant<int64, uint64, absl::string_view>;

XPlane* GetOrCreateHostXPlane(XSpace* space);

XPlane* GetOrCreateGpuXPlane(XSpace* space, int32 device_ordinal);

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats = {});

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64 offset_ps, int64 duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats = {});

void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name, int64 offset_ps,
                               int64 duration_ps,
                               absl::string_view execution_mode,
                               int64 tracing_count = -1);
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_TEST_UTILS_H_
