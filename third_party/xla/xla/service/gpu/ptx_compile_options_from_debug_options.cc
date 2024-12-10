/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/ptx_compile_options_from_debug_options.h"

#include "xla/stream_executor/cuda/compilation_options.h"

namespace xla::gpu {

stream_executor::cuda::CompilationOptions PtxCompileOptionsFromDebugOptions(
    const DebugOptions& debug_options, bool is_autotuning_compilation) {
  stream_executor::cuda::CompilationOptions compilation_options;
  compilation_options.cancel_if_reg_spill =
      (debug_options
           .xla_gpu_filter_kernels_spilling_registers_on_autotuning() &&
       is_autotuning_compilation) ||
      debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling();
  compilation_options.disable_optimizations =
      debug_options.xla_gpu_disable_gpuasm_optimizations();
  compilation_options.generate_debug_info =
      debug_options.xla_gpu_generate_debug_info();
  compilation_options.generate_line_info =
      debug_options.xla_gpu_generate_line_info();
  return compilation_options;
}
}  // namespace xla::gpu
