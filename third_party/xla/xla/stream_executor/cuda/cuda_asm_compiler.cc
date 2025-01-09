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
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cubin_or_ptx_image.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace stream_executor {

absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options) {
  return BundleGpuAsmUsingFatbin(images, options);
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

absl::StatusOr<absl::Span<const uint8_t>> CompileGpuAsmOrGetCached(
    const CudaComputeCapability& cc, const std::string& ptx,
    GpuAsmOpts compilation_options) {
  using PtxCacheKey = std::tuple<CudaComputeCapability, std::string,
                                 GpuAsmOpts::PtxOptionsTuple>;
  using PtxCompilerResult = absl::StatusOr<std::vector<uint8_t>>;
  static absl::Mutex ptx_cache_mutex(absl::kConstInit);
  static auto& ptx_cache ABSL_GUARDED_BY(ptx_cache_mutex) =
      *new absl::flat_hash_map<PtxCacheKey, PtxCompilerResult>();

  absl::MutexLock lock(&ptx_cache_mutex);
  PtxCacheKey cache_key{cc, ptx, compilation_options.ToTuple()};
  auto it = ptx_cache.find(cache_key);
  if (it == ptx_cache.end()) {
    PtxCompilerResult compiled = CompileGpuAsm(cc, ptx, compilation_options);
    it = ptx_cache.emplace(cache_key, std::move(compiled)).first;
  }

  CHECK(it != ptx_cache.end());

  // Failed compilation attempts are cached.
  // Use separate status check and ValueOrDie invocation on ptx_cache
  // entry to avoid value moving introduced by TF_ASSIGN_OR_RETURN.

  if (ABSL_PREDICT_FALSE(!it->second.ok())) {
    return it->second.status();
  }

  const std::vector<uint8_t>& compiled = it->second.value();
  return absl::MakeSpan(compiled);
}


}  // namespace stream_executor
