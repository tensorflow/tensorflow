/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_diagnostics.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/subprocess.h"

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
      return tsl::Status(absl::StatusCode::kUnknown, oss.str().c_str());      \
    }                                                                         \
  } while (false)

tsl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
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
    return tsl::errors::Internal(
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

tsl::StatusOr<std::vector<uint8_t>> LinkGpuAsm(
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

}  // namespace stream_executor
