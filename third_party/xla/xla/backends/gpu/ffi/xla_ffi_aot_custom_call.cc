/* Copyright 2026 The OpenXLA Authors.

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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cudart_kernel_registry.h"
#include "xla/stream_executor/cuda/simple_kernel_cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep

namespace xla::gpu {

struct CustomCallResources {
  stream_executor::KernelLoaderSpec spec;
};

}  // namespace xla::gpu

namespace xla::ffi {

template <>
struct TypeRegistry::SerDes<xla::gpu::CustomCallResources>
    : public std::true_type {
  static absl::StatusOr<std::string> Serialize(
      const xla::gpu::CustomCallResources& state) {
    ASSIGN_OR_RETURN(auto spec_proto, state.spec.ToProto());
    return spec_proto.SerializeAsString();
  }

  static absl::StatusOr<std::unique_ptr<xla::gpu::CustomCallResources>>
  Deserialize(absl::string_view serialized) {
    stream_executor::KernelLoaderSpecProto spec_proto;
    spec_proto.ParseFromString(serialized);
    ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec spec,
                     stream_executor::KernelLoaderSpec::FromProto(spec_proto));
    return std::make_unique<xla::gpu::CustomCallResources>(
        xla::gpu::CustomCallResources{std::move(spec)});
  }
};

}  // namespace xla::ffi

namespace xla::gpu {
namespace {

using Write42Kernel =
    stream_executor::TypedKernel<stream_executor::DeviceAddress<int32_t>,
                                 int32_t>;

using Write42KernelFactory = Write42Kernel::FactoryType;

struct CustomCallLoadedKernel {
  Write42Kernel kernel;
};

// 1. Instantiate Handler
XLA_FFI_DEFINE_HANDLER(
    kInstantiate,
    []() -> absl::StatusOr<std::unique_ptr<CustomCallResources>> {
      ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec kernel,
                       stream_executor::cuda::FindCudaRuntimeKernel(
                           stream_executor::cuda::GetWrite42Kernel()));

      return std::make_unique<CustomCallResources>(
          CustomCallResources{std::move(kernel)});
    },
    ffi::Ffi::BindInstantiate());

// 2. Initialize Handler
XLA_FFI_DEFINE_HANDLER(
    kInitialize,
    [](const CustomCallResources* resources, stream_executor::Stream* stream)
        -> absl::StatusOr<std::unique_ptr<CustomCallLoadedKernel>> {
      TF_RET_CHECK(resources != nullptr) << "CustomCallResources is null";
      TF_RET_CHECK(stream != nullptr) << "Stream is null";

      stream_executor::StreamExecutor* executor = stream->parent();
      ASSIGN_OR_RETURN(auto kernel,
                       Write42KernelFactory::Create(executor, resources->spec));
      return std::make_unique<CustomCallLoadedKernel>(
          CustomCallLoadedKernel{std::move(kernel)});
    },
    ffi::Ffi::BindInitialize()
        .Ctx<ffi::State<CustomCallResources>>()
        .Ctx<ffi::Stream>());

// 3. Execute Handler
XLA_FFI_DEFINE_HANDLER(
    kExecute,
    [](stream_executor::Stream* stream, CustomCallLoadedKernel* loaded_kernel,
       ffi::Result<ffi::Buffer<S32>> out) -> absl::Status {
      TF_RET_CHECK(stream != nullptr) << "Stream is null";
      TF_RET_CHECK(loaded_kernel != nullptr) << "Loaded kernel is null";

      return loaded_kernel->kernel.Launch(
          stream_executor::ThreadDim(),
          stream_executor::BlockDim(out->element_count()), stream,
          out->device_memory(), static_cast<int32_t>(out->element_count()));
    },
    ffi::Ffi::BindExecute()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::Initialized<CustomCallLoadedKernel>>()
        .Ret<ffi::Buffer<S32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.test_write_42_aot",
                         "CUDA",
                         {
                             /*instantiate=*/kInstantiate,
                             /*prepare=*/nullptr,
                             /*initialize=*/kInitialize,
                             /*execute=*/kExecute,
                         });

}  // namespace
}  // namespace xla::gpu
