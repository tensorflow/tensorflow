/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_GPU_METADATA_H_

#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
void AddGpuMetadata(tensorflow::profiler::XSpace* space);
}
}  // namespace xla
#define XLA_BACKENDS_PROFILER_GPU_GPU_METADATA_H_

#endif  // XLA_BACKENDS_PROFILER_GPU_GPU_METADATA_H_
