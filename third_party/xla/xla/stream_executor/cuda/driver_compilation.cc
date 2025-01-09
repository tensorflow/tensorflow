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

#include "xla/stream_executor/cuda/driver_compilation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

absl::StatusOr<std::vector<uint8_t>> LinkGpuAsmUsingDriver(
    StreamExecutor* executor, const stream_executor::CudaComputeCapability& cc,
    absl::Span<const std::vector<uint8_t>> images) {
  std::unique_ptr<ActivateContext> context = executor->Activate();

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
  for (const std::vector<uint8_t>& image : images) {
    auto status = cuda::ToStatus(cuLinkAddData(
        link_state, CU_JIT_INPUT_CUBIN, absl::bit_cast<void*>(image.data()),
        image.size(), "", 0, nullptr, nullptr));
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
  return cubin;
}

}  // namespace stream_executor
