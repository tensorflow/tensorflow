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

#include "xla/stream_executor/cuda/driver_compilation_provider.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::cuda {
absl::StatusOr<Assembly> DriverCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return CompileAndLink(cc, {Ptx{std::string{ptx}}}, options);
}

absl::StatusOr<RelocatableModule>
DriverCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  return absl::UnavailableError(
      "Compilation to relocatable module is not "
      "supported by the CUDA driver.");
}

absl::StatusOr<Assembly> DriverCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  TF_ASSIGN_OR_RETURN(Platform * platform,
                      PlatformManager::PlatformWithId(kCudaPlatformId));
  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      platform->ExecutorForDevice(0));
  std::unique_ptr<ActivateContext> context = executor->Activate();

  CUlinkState link_state;
  CUjit_option jit_options[] = {CU_JIT_TARGET,
                                CU_JIT_OPTIMIZATION_LEVEL,
                                CU_JIT_GENERATE_DEBUG_INFO,
                                CU_JIT_GENERATE_LINE_INFO,
                                CU_JIT_LOG_VERBOSE,
                                CU_JIT_INFO_LOG_BUFFER,
                                CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                CU_JIT_ERROR_LOG_BUFFER,
                                CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  CUjit_target target = static_cast<CUjit_target>(cc.major * 10 + cc.minor);
#if CUDA_VERSION >= 12000
  // Even though CUDA 11.8 has Hopper support, SM 9.0a and most Hopper features
  // (WGMMA, TMA, and more) are only supported in CUDA 12+.
  if (cc.major == 9 && cc.minor == 0) {
    target =
        static_cast<CUjit_target>(target + CU_COMPUTE_ACCELERATED_TARGET_BASE);
  }
#endif
  constexpr size_t kErrorLogBufferSize = 512 * 1024;  // 4 KiB
  std::string error_log_buffer(kErrorLogBufferSize, '\0');

  constexpr size_t kInfoLogBufferSize = 64 * 1024;  // 16 KiB
  std::string info_log_buffer(kInfoLogBufferSize, '\0');

  void* jit_option_values[] = {
      // We first cast to an integer type the same size as a pointer, and then
      // we bit cast that integer to a pointer.
      absl::bit_cast<void*>(static_cast<std::ptrdiff_t>(target)),
      absl::bit_cast<void*>(
          static_cast<std::ptrdiff_t>(options.disable_optimizations ? 0 : 4)),
      absl::bit_cast<void*>(
          static_cast<std::ptrdiff_t>(options.generate_debug_info)),
      absl::bit_cast<void*>(
          static_cast<std::ptrdiff_t>(options.generate_line_info)),
      absl::bit_cast<void*>(static_cast<std::ptrdiff_t>(
          VLOG_IS_ON(3) || options.cancel_if_reg_spill)),
      static_cast<void*>(info_log_buffer.data()),
      absl::bit_cast<void*>(
          static_cast<std::ptrdiff_t>(info_log_buffer.size())),
      static_cast<void*>(error_log_buffer.data()),
      absl::bit_cast<void*>(
          static_cast<std::ptrdiff_t>(error_log_buffer.size())),
  };

  // The buffer size fields in the jit_option_value array are also out
  // parameters that are filled by the cuLinkComplete call.
  const auto info_log_buffer_size = [&]() -> size_t {
    return absl::bit_cast<std::ptrdiff_t>(jit_option_values[6]);
  };

  const auto error_log_buffer_size = [&]() -> size_t {
    return absl::bit_cast<std::ptrdiff_t>(jit_option_values[8]);
  };

  // Both arrays must have the same number of elements.
  static_assert(sizeof(jit_options) / sizeof(jit_options[0]) ==
                sizeof(jit_option_values) / sizeof(jit_option_values[0]));

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuLinkCreate(sizeof(jit_options) / sizeof(jit_options[0]), jit_options,
                   jit_option_values, &link_state)));

  // We have to make a copy here because we need a null-terminated string.
  for (const auto& input : inputs) {
    if (std::holds_alternative<RelocatableModule>(input)) {
      const RelocatableModule& module = std::get<RelocatableModule>(input);
      TF_RETURN_IF_ERROR(cuda::ToStatus(cuLinkAddData(
          link_state, CU_JIT_INPUT_CUBIN,
          absl::bit_cast<void*>(module.cubin.data()), module.cubin.size(),
          /*name=*/"", 0, nullptr, nullptr)));
    } else {
      const std::string& ptx = std::get<Ptx>(input).ptx;
      TF_RETURN_IF_ERROR(cuda::ToStatus(cuLinkAddData(
          link_state, CU_JIT_INPUT_PTX, absl::bit_cast<void*>(ptx.data()),
          ptx.size() + 1, /*name=*/"", 0, nullptr, nullptr)));
    }
  }

  void* cubin_out;
  size_t cubin_size;
  CUresult result = cuLinkComplete(link_state, &cubin_out, &cubin_size);

  absl::Cleanup link_state_cleaner = [&link_state] {
    CHECK_EQ(cuLinkDestroy(link_state), CUDA_SUCCESS);
  };

  CHECK(error_log_buffer_size() <= kErrorLogBufferSize);
  error_log_buffer.resize(error_log_buffer_size());

  CHECK(info_log_buffer_size() <= kInfoLogBufferSize);
  info_log_buffer.resize(info_log_buffer_size());

  absl::string_view extension = (cc.major == 9 && cc.minor == 0) ? "a" : "";
  std::string architecture = absl::StrCat("sm_", cc.major, cc.minor, extension);

  if (result != CUDA_SUCCESS) {
    VLOG(3) << "Driver compilation error log output: " << error_log_buffer;
    TF_RETURN_IF_ERROR(CreateErrorFromPTXASLog(error_log_buffer, architecture,
                                               options.cancel_if_reg_spill));

    return cuda::ToStatus(result, error_log_buffer);
  }

  VLOG(3) << "Driver compilation info log output: " << info_log_buffer;
  TF_RETURN_IF_ERROR(CreateErrorFromPTXASLog(info_log_buffer, architecture,
                                             options.cancel_if_reg_spill));

  std::vector<uint8_t> cubin(static_cast<uint8_t*>(cubin_out),
                             static_cast<uint8_t*>(cubin_out) + cubin_size);
  return Assembly{std::move(cubin)};
}

}  // namespace stream_executor::cuda
