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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_FUNCTIONS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_FUNCTIONS_H_

#include <string>

#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "plugin/tensorboard_plugin_profile/protobuf/tf_function.pb.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

// Converts from the given XLine to a TfFunctionDb.
TfFunctionDb ConvertHostThreadsXLineToTfFunctionDb(const XLineVisitor& line);

// Returns a debugging string for the given TfFunctionDb.
std::string DebugString(TfFunctionDb tf_function_db);

// Combines the tf-function statistics from src and dst into dst.
void CombineTfFunctionDb(const TfFunctionDb& src, TfFunctionDb* dst);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_FUNCTIONS_H_
