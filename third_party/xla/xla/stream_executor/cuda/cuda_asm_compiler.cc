/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_asm_compiler.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cubin_or_ptx_image.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options) {
  return BundleGpuAsmUsingFatbin(images, options);
}

absl::StatusOr<std::vector<uint8_t>> LinkGpuAsm(
    stream_executor::CudaComputeCapability cc,
    std::vector<CubinOrPTXImage> images) {
  CUlinkState link_state;
  CUjit_option options[] = {CU_JIT_TARGET};
  CUjit_target target = static_cast<CUjit_target>(cc.major * 10 + cc.minor);
#if CUDA_VERSION >= 12000
  // Even though CUDA 11.8 has Hopper support, SM 9.0a and most Hopper features
  // (WGMMA, TMA, and more) are only supported in CUDA 12+.
  if (cc.major == 9 && cc.minor == 0) {
    target =
        static_cast<CUjit_target>(target + CU_COMPUTE_ACCELERATED_TARGET_BASE);
  }
#endif
  void* option_values[] = {
      // We first cast to an integer type the same size as a pointer, and then
      // we reinterpret that integer as a pointer.
      reinterpret_cast<void*>(static_cast<std::ptrdiff_t>(target))};

  // Both arrays must have the same number of elements.
  static_assert(sizeof(options) / sizeof(options[0]) ==
                sizeof(option_values) / sizeof(option_values[0]));

  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuLinkCreate(sizeof(options) / sizeof(options[0]), options,
                                  option_values, &link_state)));
  for (auto& image : images) {
    auto status = cuda::ToStatus(cuLinkAddData(
        link_state, CU_JIT_INPUT_CUBIN, static_cast<void*>(image.bytes.data()),
        image.bytes.size(), "", 0, nullptr, nullptr));
    if (!status.ok()) {
      LOG(ERROR) << "cuLinkAddData fails. This is usually caused by stale "
                    "driver version.";
      return status;
    }
  }
  void* cubin_out;
  size_t cubin_size;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuLinkComplete(link_state, &cubin_out, &cubin_size)));
  std::vector<uint8_t> cubin(static_cast<uint8_t*>(cubin_out),
                             static_cast<uint8_t*>(cubin_out) + cubin_size);
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuLinkDestroy(link_state)));
  return std::move(cubin);
}

absl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(
    const CudaComputeCapability& cc, const std::string& ptx, GpuAsmOpts options,
    bool cancel_if_reg_spill) {
  if (IsLibNvPtxCompilerSupported()) {
    VLOG(3) << "Compiling GPU ASM with libnvptxcompiler";
    return CompileGpuAsmUsingLibNvPtxCompiler(cc, ptx, options,
                                              cancel_if_reg_spill);
  }

  VLOG(3) << "Compiling GPU ASM with PTXAS. Libnvptxcompiler compilation "
             "not supported.";
  return CompileGpuAsmUsingPtxAs(cc, ptx, options, cancel_if_reg_spill);
}

}  // namespace stream_executor
