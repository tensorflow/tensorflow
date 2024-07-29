/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/gpu/kernel/tfrt_gpu_init.h"

#include <memory>
#include <utility>

#include "xla/tsl/framework/serving_device_selector_policies.h"
#include "tensorflow/core/common_runtime/gpu/gpu_serving_device_selector.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/gpu/kernel/gpu_runner.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace gpu {

Status InitTfrtGpu(const GpuRunnerOptions& options,
                   tensorflow::tfrt_stub::Runtime& runtime) {
  auto policy = std::make_unique<tsl::RoundRobinPolicy>();
  auto serving_device_selector =
      std::make_unique<tensorflow::gpu::GpuServingDeviceSelector>(
          options.num_gpu_streams, std::move(policy));

  // We need to move `serving_device_selector` to the heap here, as
  // `AddCreateRuntimeResourceFn` requires a copyable callback.
  auto shared_serving_device_selector =
      std::shared_ptr<tensorflow::gpu::GpuServingDeviceSelector>(
          serving_device_selector.release());
  runtime.AddCreateRuntimeResourceFn(
      [serving_device_selector = std::move(shared_serving_device_selector)](
          tfrt::ResourceContext* resource_ctx) mutable {
        resource_ctx->CreateResource<tensorflow::gpu::GpuRunner>(
            tensorflow::gpu::kGpuRunnerResourceName,
            serving_device_selector.get());
      });
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tensorflow
