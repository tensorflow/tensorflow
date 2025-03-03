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
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/nvJitLink.h"
#include "xla/stream_executor/cuda/nvjitlink.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

static absl::string_view ToString(nvJitLinkResult status) {
  switch (status) {
    case NVJITLINK_SUCCESS:
      return "SUCCESS";
    case NVJITLINK_ERROR_UNRECOGNIZED_OPTION:
      return "UNRECOGNIZED_OPTION";
    case NVJITLINK_ERROR_MISSING_ARCH:
      return "MISSING_ARCH";
    case NVJITLINK_ERROR_INVALID_INPUT:
      return "INVALID_INPUT";
    case NVJITLINK_ERROR_PTX_COMPILE:
      return "PTX_COMPILE";
    case NVJITLINK_ERROR_NVVM_COMPILE:
      return "NVVM_COMPILE";
    case NVJITLINK_ERROR_INTERNAL:
      return "INTERNAL";
    case NVJITLINK_ERROR_THREADPOOL:
      return "THREADPOOL";
    default:
      return "UNKNOWN";
  }
}

static absl::Status ToStatus(nvJitLinkResult status,
                             absl::string_view message) {
  return absl::UnknownError(absl::StrCat(ToString(status), ": ", message));
}

#define RETURN_IF_NVJITLINK_ERROR(expr)                                  \
  do {                                                                   \
    nvJitLinkResult _status = expr;                                      \
    if (!ABSL_PREDICT_TRUE(_status == NVJITLINK_SUCCESS)) {              \
      std::ostringstream oss;                                            \
      oss << ToString(_status) << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "'";                                     \
      return absl::UnknownError(oss.str());                              \
    }                                                                    \
  } while (false)

static absl::StatusOr<std::string> nvJitLinkGetErrorLog(
    nvJitLinkHandle link_handle) {
  size_t size{};
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkGetErrorLogSize(link_handle, &size));

  std::string error_log(size, '\0');
  RETURN_IF_NVJITLINK_ERROR(
      nvJitLinkGetErrorLog(link_handle, error_log.data()));

  return error_log;
}

static absl::StatusOr<std::string> nvJitLinkGetInfoLog(
    nvJitLinkHandle link_handle) {
  size_t size{};
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkGetInfoLogSize(link_handle, &size));

  std::string info_log(size, '\0');
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkGetInfoLog(link_handle, info_log.data()));

  return info_log;
}

extern "C" {
// We forward declare this as a weak symbol which allows us to check at runtime
// whether it's available or not (depending on the version of libnvjitlink that
// was loaded).
[[gnu::weak]] nvJitLinkResult nvJitLinkVersion(unsigned int* major,
                                               unsigned int* minor);
}

absl::StatusOr<NvJitLinkVersion> GetNvJitLinkVersion() {
  // In CUDA versions prior to 12.3 the symbol `nvJitLinkVersion` was not
  // available. So in that case we return the version 12.0 which was the first
  // CUDA version that offered libnvjitlink.
  if (!nvJitLinkVersion) {
    return NvJitLinkVersion{12, 0};
  }
  unsigned int major{}, minor{};
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkVersion(&major, &minor));
  return NvJitLinkVersion(major, minor);
}

absl::StatusOr<std::vector<uint8_t>> CompileAndLinkUsingLibNvJitLink(
    const CudaComputeCapability& cc, absl::Span<const NvJitLinkInput> inputs,
    GpuAsmOpts options, bool cancel_if_reg_spill) {
  if (inputs.empty()) {
    return std::vector<uint8_t>();
  }

  TF_ASSIGN_OR_RETURN(NvJitLinkVersion version, GetNvJitLinkVersion());
  auto [version_major, version_minor] = version;
  WarnIfBadPtxasVersion("nvJitLink", cc, {version_major, version_minor, 0});

  std::vector<std::string> cli_args;
  // On Hopper, default to sm_90a so that all instructions can be used. But
  // only sm_90 is forward compatible, so don't use sm_90a with newer hardware:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility
  absl::string_view extension = ShouldUsePtxExtension(cc) ? "a" : "";
  std::string architecture = absl::StrCat("sm_", cc.major, cc.minor, extension);
  cli_args.emplace_back(absl::StrCat("-arch=", architecture));

  if (VLOG_IS_ON(2)) {
    cli_args.emplace_back("-verbose");
  }
  cli_args.emplace_back("-Xptxas=--warn-on-spills");
  cli_args.emplace_back(absl::StrCat("-split-compile=", inputs.size()));

  if (options.disable_gpuasm_optimizations) {
    cli_args.emplace_back("-Xptxas=-O0");
  }

  absl::c_transform(
      options.extra_flags, std::back_inserter(cli_args),
      [](const std::string& s) { return absl::StrCat("-Xptxas=", s); });

  VLOG(2) << "nvJitLink options: " << absl::StrJoin(cli_args, " ");

  std::vector<const char*> cli_args_ptrs{};
  absl::c_transform(cli_args, std::back_inserter(cli_args_ptrs),
                    [](const std::string& s) { return s.c_str(); });

  nvJitLinkHandle link_handle{};
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkCreate(&link_handle, cli_args_ptrs.size(),
                                            cli_args_ptrs.data()));
  absl::Cleanup link_handle_cleaner = [&link_handle] {
    CHECK_EQ(nvJitLinkDestroy(&link_handle), NVJITLINK_SUCCESS);
  };

  for (auto& image : inputs) {
    nvJitLinkInputType input_type = image.type == NvJitLinkInput::Type::kPtx
                                        ? NVJITLINK_INPUT_PTX
                                        : NVJITLINK_INPUT_CUBIN;
    // When the input type is PTX, then `nvJitLinkAddData` ignores the size
    // argument and expects a null-terminated string as the data input. So we
    // make sure that our input ends with a null byte. (Don't ask me how I
    // know.)
    if (input_type == NVJITLINK_INPUT_PTX && !image.bytes.empty()) {
      CHECK_EQ(image.bytes.back(), '\0');
    }

    nvJitLinkResult result =
        nvJitLinkAddData(link_handle, input_type, image.bytes.data(),
                         image.bytes.size(), nullptr);

    if (result != NVJITLINK_SUCCESS) {
      TF_ASSIGN_OR_RETURN(std::string error_log,
                          nvJitLinkGetErrorLog(link_handle));

      // Print the verbose output of ptxas.
      VLOG(3) << "libnvjitlink error log output: " << error_log;

      TF_RETURN_IF_ERROR(CreateErrorFromPTXASLog(error_log, architecture,
                                                 cancel_if_reg_spill));
      return ToStatus(result, error_log);
    }
  }

  nvJitLinkResult linking_result = nvJitLinkComplete(link_handle);

  if (linking_result != NVJITLINK_SUCCESS) {
    TF_ASSIGN_OR_RETURN(std::string error_log,
                        nvJitLinkGetErrorLog(link_handle));

    // Print the verbose output of ptxas.
    VLOG(3) << "libnvjitlink error log output: " << error_log;

    TF_RETURN_IF_ERROR(
        CreateErrorFromPTXASLog(error_log, architecture, cancel_if_reg_spill));
    return ToStatus(linking_result, error_log);
  }

  TF_ASSIGN_OR_RETURN(std::string info_log, nvJitLinkGetInfoLog(link_handle));

  // Print the verbose output of ptxas.
  VLOG(3) << "libnvjitlink info log output: " << info_log;

  TF_RETURN_IF_ERROR(
      CreateErrorFromPTXASLog(info_log, architecture, cancel_if_reg_spill));

  size_t cubin_size{};
  RETURN_IF_NVJITLINK_ERROR(
      nvJitLinkGetLinkedCubinSize(link_handle, &cubin_size));

  std::vector<uint8_t> cubin(cubin_size);
  RETURN_IF_NVJITLINK_ERROR(nvJitLinkGetLinkedCubin(link_handle, cubin.data()));

  return cubin;
}
#undef RETURN_IF_NVJITLINK_ERROR

}  // namespace stream_executor
