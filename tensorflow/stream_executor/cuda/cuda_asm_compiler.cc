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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/gpu_diagnostics.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {

#define RETURN_IF_CUDA_ERROR(expr)                                            \
  do {                                                                        \
    CUresult _status = expr;                                                  \
    if (!SE_PREDICT_TRUE(_status == CUDA_SUCCESS)) {                          \
      const char* error_string;                                               \
      cuGetErrorString(_status, &error_string);                               \
      std::ostringstream oss;                                                 \
      oss << error_string << "\nin " << __FILE__ << "(" << __LINE__ << "): '" \
          << #expr << "'";                                                    \
      return port::Status(port::error::UNKNOWN, oss.str().c_str());           \
    }                                                                         \
  } while (false)

port::StatusOr<std::vector<uint8>> LinkGpuAsm(
    gpu::GpuContext* context, std::vector<CubinOrPTXImage> images) {
  if (CUDA_VERSION >= 11030) {
    int driver_cuda_version;
    // Get the highest version of CUDA supported by this driver.
    RETURN_IF_CUDA_ERROR(cuDriverGetVersion(&driver_cuda_version));
    if (driver_cuda_version < CUDA_VERSION) {
      return tensorflow::errors::Unimplemented(
          "CUDA version unsupported by NVIDIA driver version.");
    }
  }

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
  std::vector<uint8> cubin(static_cast<uint8*>(cubin_out),
                           static_cast<uint8*>(cubin_out) + cubin_size);
  RETURN_IF_CUDA_ERROR(cuLinkDestroy(link_state));
  return std::move(cubin);
}

}  // namespace stream_executor
