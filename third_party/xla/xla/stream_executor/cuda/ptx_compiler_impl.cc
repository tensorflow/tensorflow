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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/nvPTXCompiler.h"
#include "xla/stream_executor/cuda/ptx_compiler.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

static std::string_view ToString(nvPTXCompileResult status) {
  switch (status) {
    case NVPTXCOMPILE_SUCCESS:
      return "SUCCESS";
    case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
      return "INVALID_COMPILER_HANDLE";
    case NVPTXCOMPILE_ERROR_INVALID_INPUT:
      return "INVALID_INPUT";
    case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
      return "COMPILATION_FAILURE";
    case NVPTXCOMPILE_ERROR_INTERNAL:
      return "INTERNAL";
    case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
      return "COMPILER_INVOCATION_INCOMPLETE";
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
      return "UNSUPPORTED_PTX_VERSION";
#if CUDA_VERSION > 12000
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC:
      return "UNSUPPORTED_DEVSIDE_SYNC";
#endif
    default:
      return "UNKNOWN";
  }
}

#define RETURN_IF_NVPTXCOMPILER_ERROR(expr)                              \
  do {                                                                   \
    nvPTXCompileResult _status = expr;                                   \
    if (!ABSL_PREDICT_TRUE(_status == NVPTXCOMPILE_SUCCESS)) {           \
      std::ostringstream oss;                                            \
      oss << ToString(_status) << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "'";                                     \
      return absl::UnknownError(oss.str());                              \
    }                                                                    \
  } while (false)

absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingLibNvPtxCompiler(
    const CudaComputeCapability& cc, const std::string& ptx_contents,
    GpuAsmOpts options, bool cancel_if_reg_spill) {
  TF_ASSIGN_OR_RETURN(auto version, GetLibNvPtxCompilerVersion());
  WarnIfBadPtxasVersion("nvPTXCompiler", cc, version);

  nvPTXCompilerHandle compiler_handle{};
  RETURN_IF_NVPTXCOMPILER_ERROR(nvPTXCompilerCreate(
      &compiler_handle, ptx_contents.size(), ptx_contents.data()));
  absl::Cleanup compiler_cleaner = [&compiler_handle] {
    nvPTXCompilerDestroy(&compiler_handle);
  };
  // On Hopper, default to sm_90a so that all instructions can be used. But
  // only sm_90 is forward compatible, so don't use sm_90a with newer hardware:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility
  std::string_view extension = (cc.major == 9 && cc.minor == 0) ? "a" : "";
  std::string architecture = absl::StrCat("sm_", cc.major, cc.minor, extension);

  options.extra_flags.emplace_back(absl::StrCat("-arch=", architecture));
  options.extra_flags.emplace_back("--warn-on-spills");

  if (VLOG_IS_ON(2)) {
    options.extra_flags.emplace_back("-v");
  }
  if (options.disable_gpuasm_optimizations) {
    options.extra_flags.emplace_back("-O0");
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << absl::StrJoin(options.extra_flags, " ");
  }

  std::vector<const char*> cmdline_options_ptrs{};
  absl::c_transform(options.extra_flags,
                    std::back_inserter(cmdline_options_ptrs),
                    [](const std::string& s) { return s.c_str(); });

  nvPTXCompileResult compile_result =
      nvPTXCompilerCompile(compiler_handle, cmdline_options_ptrs.size(),
                           cmdline_options_ptrs.data());

  if (compile_result != NVPTXCOMPILE_SUCCESS) {
    size_t error_log_size{};
    RETURN_IF_NVPTXCOMPILER_ERROR(
        nvPTXCompilerGetErrorLogSize(compiler_handle, &error_log_size));

    std::string error_log(error_log_size, '\0');
    RETURN_IF_NVPTXCOMPILER_ERROR(
        nvPTXCompilerGetErrorLog(compiler_handle, error_log.data()));

    //  It happens when the linked version of ntvptxcompiler is too old for the
    //  current GPU. Example error message associated with this error code:
    //      ptxas fatal   : Value 'sm_80' is not defined for option 'gpu-name'
    if (absl::StrContains(error_log, "ptxas fatal   : Value '") &&
        absl::StrContains(error_log, "is not defined for option 'gpu-name'")) {
      return absl::UnimplementedError(absl::StrFormat(
          "Linked libnvptxcompiler is too old for %s.", architecture));
    }
    if (IsPtxRegisterAllocationError(error_log)) {
      return absl::ResourceExhaustedError(error_log);
    }

    return absl::InternalError(
        absl::StrFormat("PTX compilation failed with error code %d, output: %s",
                        compile_result, error_log));
  }

  size_t info_log_size{};
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetInfoLogSize(compiler_handle, &info_log_size));

  std::vector<char> info_log_buffer(info_log_size + 1);
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetInfoLog(compiler_handle, info_log_buffer.data()));
  // The buffer may have several trailing null characters, so create a string
  // from the pointer to the buffer rather than pair of iterators.
  std::string info_log(info_log_buffer.data());

  // Print the verbose output of ptxas.
  if (!info_log.empty()) {
    if (absl::StrContains(info_log, "warning")) {
      LOG(INFO) << info_log;
      if (cancel_if_reg_spill &&
          absl::StrContains(info_log, "Registers are spilled")) {
        return absl::CancelledError(
            "Compilation result discarded due to register spilling");
      }
    } else {
      VLOG(2) << info_log;
    }
  }

  size_t cubinSize{};
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetCompiledProgramSize(compiler_handle, &cubinSize));

  std::vector<uint8_t> cubin(cubinSize);
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetCompiledProgram(compiler_handle, (char*)cubin.data()));

  return cubin;
}

absl::StatusOr<SemanticVersion> GetLibNvPtxCompilerVersion() {
  unsigned major{}, minor{};
  RETURN_IF_NVPTXCOMPILER_ERROR(nvPTXCompilerGetVersion(&major, &minor));

  return SemanticVersion{major, minor, 0};
}

}  // namespace stream_executor
