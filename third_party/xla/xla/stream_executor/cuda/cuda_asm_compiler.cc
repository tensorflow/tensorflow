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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/subprocess.h"

#ifdef ENABLE_LIBNVPTXCOMPILER_SUPPORT
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/nvPTXCompiler.h"
#endif

namespace stream_executor {

#define RETURN_IF_CUDA_ERROR(expr)                                            \
  do {                                                                        \
    CUresult _status = expr;                                                  \
    if (!ABSL_PREDICT_TRUE(_status == CUDA_SUCCESS)) {                        \
      const char* error_string;                                               \
      cuGetErrorString(_status, &error_string);                               \
      std::ostringstream oss;                                                 \
      oss << error_string << "\nin " << __FILE__ << "(" << __LINE__ << "): '" \
          << #expr << "'";                                                    \
      return absl::UnknownError(oss.str().c_str());                           \
    }                                                                         \
  } while (false)

absl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
    absl::string_view preferred_cuda_dir, gpu::GpuContext* context,
    std::vector<CubinOrPTXImage> images) {
  {
    static absl::once_flag log_once;
    absl::call_once(log_once,
                    [] { LOG(INFO) << "Using nvlink for parallel linking"; });
  }
  const std::string bin_path =
      FindCudaExecutable("nvlink", std::string(preferred_cuda_dir));

  if (images.empty()) {
    return std::vector<uint8>();
  }

  auto env = tsl::Env::Default();
  std::vector<std::string> temp_files;
  absl::Cleanup cleaners = [&] {
    for (auto& f : temp_files) {
      TF_CHECK_OK(tsl::Env::Default()->DeleteFile(f));
    }
  };
  for (int i = 0; i < images.size(); i++) {
    temp_files.emplace_back();
    TF_RET_CHECK(env->LocalTempFilename(&temp_files.back()));
    temp_files.back() += ".cubin";
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
        env, temp_files.back(),
        absl::string_view(reinterpret_cast<const char*>(images[i].bytes.data()),
                          images[i].bytes.size())));
  }
  std::string output_path;
  TF_RET_CHECK(env->LocalTempFilename(&output_path));
  absl::Cleanup output_cleaner = [&] {
    // CUBIN file may never be created, so the failure to delete it should not
    // produce TF error.
    tsl::Env::Default()->DeleteFile(output_path).IgnoreError();
  };
  int cc_major;
  int cc_minor;
  {
    TF_ASSIGN_OR_RETURN(auto cu_device,
                        gpu::GpuDriver::DeviceFromContext(context));
    TF_RETURN_IF_ERROR(
        gpu::GpuDriver::GetComputeCapability(&cc_major, &cc_minor, cu_device));
  }
  std::vector<std::string> args;
  args.push_back(bin_path);
  args.push_back(absl::StrCat("-arch=sm_", cc_major, cc_minor));
  for (int i = 0; i < images.size(); i++) {
    args.push_back(temp_files[i]);
  }
  args.push_back("-o");
  args.push_back(output_path);

  tsl::SubProcess process;
  process.SetProgram(bin_path, args);
  process.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  TF_RET_CHECK(process.Start());
  std::string stderr_output;
  int exit_status = process.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);

  if (exit_status != 0) {
    return absl::InternalError(
        absl::StrFormat("nvlink exited with non-zero error code %d, output: %s",
                        exit_status, stderr_output));
  }

  if (!stderr_output.empty()) {
    if (absl::StrContains(stderr_output, "warning")) {
      LOG(INFO) << stderr_output;
    } else {
      VLOG(2) << stderr_output;
    }
  }

  // Read in the result of compilation and return it as a byte vector.
  std::string cubin;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), output_path, &cubin));
  std::vector<uint8_t> cubin_vector(cubin.begin(), cubin.end());
  return cubin_vector;
}

absl::StatusOr<std::vector<uint8_t>> LinkGpuAsm(
    gpu::GpuContext* context, std::vector<CubinOrPTXImage> images) {
  gpu::ScopedActivateContext activation(context);

  CUlinkState link_state;
  RETURN_IF_CUDA_ERROR(cuLinkCreate(0, nullptr, nullptr, &link_state));
  for (auto& image : images) {
    auto status = cuLinkAddData(link_state, CU_JIT_INPUT_CUBIN,
                                static_cast<void*>(image.bytes.data()),
                                image.bytes.size(), "", 0, nullptr, nullptr);
    if (status != CUDA_SUCCESS) {
      LOG(ERROR) << "cuLinkAddData fails. This is usually caused by stale "
                    "driver version.";
    }
    RETURN_IF_CUDA_ERROR(status);
  }
  void* cubin_out;
  size_t cubin_size;
  RETURN_IF_CUDA_ERROR(cuLinkComplete(link_state, &cubin_out, &cubin_size));
  std::vector<uint8_t> cubin(static_cast<uint8_t*>(cubin_out),
                             static_cast<uint8_t*>(cubin_out) + cubin_size);
  RETURN_IF_CUDA_ERROR(cuLinkDestroy(link_state));
  return std::move(cubin);
}

#ifdef ENABLE_LIBNVPTXCOMPILER_SUPPORT
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
    int cc_major, int cc_minor, const char* ptx_contents, GpuAsmOpts options,
    bool cancel_if_reg_spill) {
  nvPTXCompilerHandle compiler_handle{};
  RETURN_IF_NVPTXCOMPILER_ERROR(nvPTXCompilerCreate(
      &compiler_handle, std::strlen(ptx_contents), ptx_contents));
  absl::Cleanup compiler_cleaner = [&compiler_handle] {
    nvPTXCompilerDestroy(&compiler_handle);
  };

  // If the target is sm_90, hard code it to sm_90a so that all instructions
  // can be used. We don't need the portability that sm_90 gives.
  std::string_view extension = (cc_major == 9 && cc_minor == 0) ? "a" : "";
  std::string architecture = absl::StrCat("sm_", cc_major, cc_minor, extension);

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
    if (absl::StrContains(error_log, "ptxas fatal") &&
        absl::StrContains(error_log, "Register allocation failed")) {
      return absl::ResourceExhaustedError("Register allocation failed");
    }

    return absl::InternalError(
        absl::StrFormat("PTX compilation failed with error code %d, output: %s",
                        compile_result, error_log));
  }

  size_t info_log_size{};
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetInfoLogSize(compiler_handle, &info_log_size));

  std::string info_log(info_log_size, '\0');
  RETURN_IF_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetInfoLog(compiler_handle, info_log.data()));

  // Print the verbose output of ptxas.
  if (!info_log.empty()) {
    if (absl::StrContains(info_log, "warning")) {
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

#undef RETURN_IF_NVPTXCOMPILER_ERROR
#endif
}  // namespace stream_executor
