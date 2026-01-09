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

#ifndef XLA_SERVICE_GPU_TESTS_HLO_PJRT_GPU_TEST_BASE_H_
#define XLA_SERVICE_GPU_TESTS_HLO_PJRT_GPU_TEST_BASE_H_

#include <memory>

#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"

namespace xla::gpu {

class HloPjRtGpuTestBase : public HloRunnerAgnosticTestBase {
 protected:
  explicit HloPjRtGpuTestBase(HloPjRtTestBaseOptions options = {});

  const GpuTargetConfig& gpu_target_config() const {
    return gpu_target_config_;
  }

  const stream_executor::DeviceDescription& device_description() const {
    return gpu_target_config_.device_description;
  }

  Compiler* compiler() const { return compiler_.get(); }

 private:
  HloPjRtGpuTestBase(PjRtClient* client, HloPjRtTestBaseOptions options);
  HloPjRtGpuTestBase(DeviceShapeRepresentationFn device_shape_representation_fn,
                     DeviceShapeSizeFn device_shape_size_fn,
                     GpuTargetConfig gpu_target_config,
                     std::unique_ptr<PjRtClient> client,
                     HloPjRtTestBaseOptions options);

  GpuTargetConfig gpu_target_config_;

  std::unique_ptr<Compiler> compiler_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TESTS_HLO_PJRT_GPU_TEST_BASE_H_
