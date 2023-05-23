/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_

#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Preprocess XSpaces before tools conversion.
// If step_grouping = true, perform events grouping for step tracking.
// If derived_timeline, generate derived timeline (XLines).
void PreprocessSingleHostXSpace(XSpace* space, bool step_grouping,
                                bool derived_timeline);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_
