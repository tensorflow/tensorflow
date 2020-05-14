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

#include "tensorflow/core/profiler/utils/errors.h"

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kErrorIncompleteStep =
    "Incomplete step observed and hence the step time is unknown."
    "Instead, we use the trace duration as the step time. This may happen"
    " if your profiling duration is shorter than the step time. In this"
    " case, you may try to profile longer.";

const absl::string_view kErrorNoStepMarker =
    "No step marker observed and hence the step time is unknown."
    " This may happen if (1) training steps are not instrumented (e.g., if"
    " you are not using Keras) or (2) the profiling duration is shorter"
    " than the step time. For (1), you need to add step instrumentation;"
    " for (2), you may try to profile longer.";

}  // namespace profiler
}  // namespace tensorflow
