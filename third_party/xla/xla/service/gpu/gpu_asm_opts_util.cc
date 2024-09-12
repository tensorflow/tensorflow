/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_asm_opts_util.h"

#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

stream_executor::GpuAsmOpts PtxOptsFromDebugOptions(
    const DebugOptions& debug_options) {
  std::vector<std::string> extra_flags = absl::StrSplit(
      debug_options.xla_gpu_asm_extra_flags(), ',', absl::SkipEmpty());
  return stream_executor::GpuAsmOpts(
      debug_options.xla_gpu_disable_gpuasm_optimizations(),
      debug_options.xla_gpu_cuda_data_dir(), extra_flags);
}

}  // namespace gpu
}  // namespace xla
