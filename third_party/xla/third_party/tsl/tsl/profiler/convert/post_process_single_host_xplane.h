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
#ifndef TENSORFLOW_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_
#define TENSORFLOW_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_

#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Post process XSpaces collected locally from multiple profilers.
void PostProcessSingleHostXSpace(tensorflow::profiler::XSpace* space,
                                 uint64 start_time_ns, uint64 stop_time_ns);

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_
