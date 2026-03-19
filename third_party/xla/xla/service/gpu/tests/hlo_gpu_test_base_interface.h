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

#ifndef XLA_SERVICE_GPU_TESTS_HLO_GPU_TEST_BASE_INTERFACE_H_
#define XLA_SERVICE_GPU_TESTS_HLO_GPU_TEST_BASE_INTERFACE_H_

#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

class HloGpuTestBaseInterface {
 protected:
  HloGpuTestBaseInterface() = default;
  virtual ~HloGpuTestBaseInterface() = default;

  virtual const GpuTargetConfig& gpu_target_config() const = 0;
  virtual const stream_executor::DeviceDescription& device_description()
      const = 0;
  virtual Compiler* compiler() const = 0;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TESTS_HLO_GPU_TEST_BASE_INTERFACE_H_
