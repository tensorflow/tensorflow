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

#include "tensorflow/compiler/xla/service/gpu/runtime/memcpy.h"

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::StridedMemrefView;

namespace {

enum class MemcpyDirection { kDeviceToDevice, kDeviceToHost, kHostToDevice };

template <MemcpyDirection direction>
struct Memcpy {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::StridedMemrefView dst,
                          runtime::StridedMemrefView src) const;
  static Memcpy Handler() { return Memcpy(); }
};
}  // namespace

template <MemcpyDirection direction>
absl::Status Memcpy<direction>::operator()(
    const ServiceExecutableRunOptions* run_options,
    runtime::StridedMemrefView dst, runtime::StridedMemrefView src) const {
  se::Stream* stream = run_options->stream();

  if (dst.sizes != src.sizes) {
    return absl::InvalidArgumentError(
        "Source memref sizes do not match destination memref sizes");
  }

  if (dst.strides != src.strides) {
    return absl::InvalidArgumentError(
        "Source memref strides do not match destination memref strides");
  }

  switch (direction) {
    case MemcpyDirection::kDeviceToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(&dst_data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kDeviceToHost: {
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(dst.data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kHostToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      stream->ThenMemcpy(&dst_data, src.data, dst_data.size());
    } break;
  }

  // TODO(jacksonstokes): H2D and D2H memcpy instead of blocking the execution
  // thread should return an async token that will become available when
  // transfer is completed.
  if (direction != MemcpyDirection::kDeviceToDevice) {
    auto st = stream->BlockHostUntilDone();
    if (!st.ok()) return ToAbslStatus(st);
  }

  return absl::OkStatus();
}

template <MemcpyDirection direction>
static bool MemcpyFn(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // dst
                             .Arg<runtime::StridedMemrefView>()  // src
                             .To<checks>(Memcpy<direction>::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterMemcpyCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.memcpy.d2d",
                    &MemcpyFn<MemcpyDirection::kDeviceToDevice>);
  registry.Register("xla.gpu.memcpy.h2d",
                    &MemcpyFn<MemcpyDirection::kHostToDevice>);
  registry.Register("xla.gpu.memcpy.d2h",
                    &MemcpyFn<MemcpyDirection::kDeviceToHost>);
}

}  // namespace gpu
}  // namespace xla
