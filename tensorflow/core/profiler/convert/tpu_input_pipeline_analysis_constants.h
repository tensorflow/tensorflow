/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TPU_INPUT_PIPELINE_ANALYSIS_CONSTANTS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TPU_INPUT_PIPELINE_ANALYSIS_CONSTANTS_H_

#include "absl/strings/string_view.h"
#include "tsl/platform/macros.h"

namespace tensorflow {
namespace profiler {

TF_CONST_INIT extern const absl::string_view kProfileAllHostsDoc;
TF_CONST_INIT extern const absl::string_view kSparseCoreV0Name;

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TPU_INPUT_PIPELINE_ANALYSIS_CONSTANTS_H_
