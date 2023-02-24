/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/custom_call.h"

#include <string>
#include <string_view>

#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/triangular_solve.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

// Custom calls with API version API_VERSION_TYPED_FFI lowered directly to an
// Xla runtime custom calls. Older API versions handled by adapting Xla runtime
// calling convention to the calling convention expected by the registered
// handler.
//
// Once all Xla backends will use Xla runtime we will deprecate older API
// version, and migrate all users to API_VERSION_TYPED_FFI.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using xla::runtime::CustomCall;
using xla::runtime::FlatMemrefView;
using xla::runtime::StridedMemrefView;

static absl::Status XlaCustomCallImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CustomCall::RemainingArgs args,
    std::string_view call_target_name, int32_t api_version,
    std::string_view backend_config) {
  // Pattern match custom call to a few special cases, otherwise find the custom
  // call handler regustered with the runtime.
  if (call_target_name == kTriangularSolveCallTarget)
    return TriangularSolve::run(run_options, debug_options, args,
                                backend_config);

  // Find the Xla custom call handler.
  auto& platform_name = run_options->stream()->parent()->platform()->Name();
  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      std::string(call_target_name), platform_name);
  if (!call_target) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot find the Xla custom call handler ", call_target_name));
  }

  // Prepare pointers to buffers to pass to the Xla custom call handler.
  llvm::SmallVector<void*> buffers;
  for (unsigned i = 0; i < args.size(); ++i) {
    if (auto memref = args.get<FlatMemrefView>(i); succeeded(memref)) {
      buffers.push_back(memref->data);
      continue;
    }

    if (auto strided = args.get<StridedMemrefView>(i); succeeded(strided)) {
      buffers.push_back(strided->data);
      continue;
    }

    // TODO(ezhulenev): Add dialect and type to model Xla custom call holes,
    // today we rely on the fact that custom calls do not support scalar
    // arguments and we can disambiguate holes from real arguments.
    if (auto hole = args.get<int64_t>(i); succeeded(hole)) {
      buffers.push_back(nullptr);
      continue;
    }

    return absl::InvalidArgumentError(
        "Failed to get arguments as (strided) memref view");
  }

  // Call custom call handler using the calling convention it requires.
  using ApiVersion = CustomCallApiVersion;

  // Original custom call API version that doesn't support returning status.
  if (api_version == ApiVersion::API_VERSION_ORIGINAL) {
    using XlaCustomCallType =
        void (*)(se::gpu::GpuStreamHandle, void**, const char*, size_t);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size());

    return absl::OkStatus();
  }

  // Xla Custom call API returning status.
  if (api_version == ApiVersion::API_VERSION_STATUS_RETURNING ||
      api_version == ApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED) {
    using XlaCustomCallType =
        void (*)(se::gpu::GpuStreamHandle, void**, const char*, size_t,
                 XlaCustomCallStatus*);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    XlaCustomCallStatus custom_call_status;
    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size(), &custom_call_status);

    if (auto message = CustomCallStatusGetMessage(&custom_call_status)) {
      return absl::InternalError(message.value());
    } else {
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported custom call API version: %d", api_version));
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    XlaCustomCall, FunctionWrapper<XlaCustomCallImpl>(), checks,
    runtime::CustomCall::Bind("xla.gpu.memcpy")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .Arg<CustomCall::RemainingArgs>()  // args
        .Attr<std::string_view>("call_target_name")
        .Attr<int32_t>("api_version")
        .Attr<std::string_view>("backend_config"));

void RegisterXlaClassicCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.custom_call", XlaCustomCall);
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace gpu
}  // namespace xla
