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

#ifndef XLA_SERVICE_GPU_GPU_ASM_OPTS_UTIL_H_
#define XLA_SERVICE_GPU_GPU_ASM_OPTS_UTIL_H_

#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Create GpuAsmOpts out of DebugOptions.
stream_executor::GpuAsmOpts PtxOptsFromDebugOptions(
    const DebugOptions& debug_options);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_ASM_OPTS_UTIL_H_
