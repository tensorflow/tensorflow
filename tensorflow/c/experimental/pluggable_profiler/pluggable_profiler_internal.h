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
#ifndef TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Plugin initialization function that a device plugin must define. Returns
// a TF_Status output specifying whether the initialization is successful.
using TFInitProfilerFn = void (*)(TF_ProfilerRegistrationParams* const,
                                  TF_Status* const);

// Registers plugin's profiler to TensorFlow's profiler registry.
Status InitPluginProfiler(TFInitProfilerFn init_fn);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_INTERNAL_H_
