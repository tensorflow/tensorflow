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
#ifndef XLA_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_
#define XLA_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_

#include "xla/tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Aligns Mosaic GPU on-device traces with corresponding kernel execution events
// on the host. This function finds pairs of Mosaic launch and dump events on
// the CPU, correlates them with CUPTI kernel launch events, and then uses the
// correlation ID to find the actual kernel execution times on the GPU. The
// timestamps of the Mosaic on-device trace plane are then adjusted accordingly.
// This should be called before MergeHostPlanes.
void AlignMosaicGpuOndeviceTrace(tensorflow::profiler::XSpace* space);

// Post process XSpaces collected locally from multiple profilers.
void PostProcessSingleHostXSpace(tensorflow::profiler::XSpace* space,
                                 uint64_t start_time_ns, uint64_t stop_time_ns);

void MergePlanesWithSameNames(tensorflow::profiler::XSpace* space);

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_CONVERT_POST_PROCESS_SINGLE_HOST_XPLANE_H_
