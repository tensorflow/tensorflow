/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

GpuClientOptions GetTestGpuClientOptions(int num_devices) {
  GpuClientOptions options;
  std::set<int> allowed;
  for (int i = 0; i < num_devices; ++i) {
    allowed.insert(i);
  }
  options.allowed_devices = std::move(allowed);
  return options;
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options) {
  ASSIGN_OR_RETURN(auto hlo_module,
                   ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

absl::StatusOr<std::shared_ptr<xla::Literal>> ExtractSingleResult(
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>&
        result) {
  ASSIGN_OR_RETURN(auto result_vec, std::move(result));
  TF_RET_CHECK(result_vec.size() == 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result_vec[0];
  TF_RET_CHECK(result_buffers.size() == 1);
  auto literal_or = result_buffers[0]->ToLiteral().Await();
  if (!literal_or.status().ok()) {
    return literal_or.status();
  }
  return *literal_or;
}

static absl::Status MemsetFromValue(
    se::Stream* stream, ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result,
    MemsetValue* memset_value) {
  uint32_t pattern;
  std::memcpy(&pattern, &memset_value->value, sizeof(pattern));

  se::DeviceAddressBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromValue, MemsetFromValue,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>()
                           .Ctx<ffi::UserData<MemsetValue>>());

std::string GetGpuPlatformName() {
  auto name = PlatformUtil::CanonicalPlatformName("GPU");
  return name.ok() ? *name : "gpu";
}

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromValue",
                         GetGpuPlatformName(), kMemsetFromValue);

static absl::Status MemsetFromAttr(
    se::Stream* stream, float attr,
    ffi::Result<ffi::BufferR1<PrimitiveType::F32>> result) {
  uint32_t pattern;
  std::memcpy(&pattern, &attr, sizeof(pattern));

  se::DeviceAddressBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetFromAttr, MemsetFromAttr,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Attr<float>("attr")
                           .Ret<ffi::BufferR1<PrimitiveType::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "MemsetFromAttr",
                         GetGpuPlatformName(), kMemsetFromAttr);

absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateDeviceBufferForTest(
    xla::PjRtClient* client) {
  auto device = client->addressable_devices()[0];
  ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  ASSIGN_OR_RETURN(
      auto input, client->BufferFromHostBuffer(
                      data.data(), shape.element_type(), shape.dimensions(),
                      /*byte_strides=*/std::nullopt,
                      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr, memory_space,
                      /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");
  return input;
}

}  // namespace xla
