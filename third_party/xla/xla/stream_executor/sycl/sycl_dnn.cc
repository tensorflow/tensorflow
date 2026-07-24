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

#include "xla/stream_executor/sycl/sycl_dnn.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dnnl.hpp"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {

OnednnSupport::OnednnSupport(StreamExecutor* parent) : parent_(parent) {}

absl::Status OnednnSupport::Init() { return absl::OkStatus(); }

absl::StatusOr<dnn::VersionInfo> OnednnSupport::GetOnednnVersion() {
  const dnnl_version_t* v = dnnl::version();
  if (v == nullptr) {
    return absl::InternalError("Failed to query oneDNN version.");
  }
  return dnn::VersionInfo(v->major, v->minor, v->patch);
}

absl::StatusOr<dnn::VersionInfo> OnednnSupport::GetVersion() {
  return GetOnednnVersion();
}

absl::Status OnednnSupport::DoConvolveWithGpuConfig(
    Stream* stream, const xla::gpu::GpuConvConfig& config,
    absl::Span<const DeviceMemoryBase> operand_se_buffers,
    DeviceMemoryBase result_se_buffer, ScratchAllocator* scratch_allocator) {
  return absl::UnimplementedError(
      "DoConvolveWithGpuConfig is not implemented for SYCL oneDNN");
}

void initialize_onednn() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          sycl::kSyclPlatformId, "oneDNN",
          [](StreamExecutor* parent) -> dnn::DnnSupport* {
            sycl::OnednnSupport* dnn = new sycl::OnednnSupport(parent);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register oneDNN factory: " << status.message();
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_onednn, {
  stream_executor::sycl::initialize_onednn();
});
