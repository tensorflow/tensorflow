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

#ifndef XLA_SERVICE_GPU_TESTS_HLO_LEGACY_GPU_TEST_BASE_H_
#define XLA_SERVICE_GPU_TESTS_HLO_LEGACY_GPU_TEST_BASE_H_

#include "absl/base/attributes.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/tests/hlo_gpu_test_base_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class ABSL_DEPRECATED(
    "Please use HloPjRtTestBase or HloPjRtGpuTestBase instead.")
    HloLegacyGpuTestBase : public HloTestBase,
                           public HloGpuTestBaseInterface {
 protected:
  explicit HloLegacyGpuTestBase(
      bool verifier_layout_sensitive = false,
      bool allow_mixed_precision_in_hlo_verifier = true,
      HloPredicate instruction_can_change_layout_func = {})
      : HloTestBase(verifier_layout_sensitive,
                    allow_mixed_precision_in_hlo_verifier,
                    instruction_can_change_layout_func) {}

  HloLegacyGpuTestBase(se::Platform* test_platform,
                       se::Platform* reference_platform,
                       bool verifier_layout_sensitive = false,
                       bool allow_mixed_precision_in_hlo_verifier = true,
                       HloPredicate instruction_can_change_layout_func = {})
      : HloTestBase(test_platform, reference_platform,
                    verifier_layout_sensitive,
                    allow_mixed_precision_in_hlo_verifier,
                    instruction_can_change_layout_func) {}

  const GpuTargetConfig& gpu_target_config() const override {
    return gpu_target_config_;
  }

  const stream_executor::DeviceDescription& device_description()
      const override {
    return gpu_target_config_.device_description;
  }

  Compiler* compiler() const override { return backend().compiler(); }

 private:
  GpuTargetConfig gpu_target_config_ =
      Compiler::GpuTargetConfig(backend().default_stream_executor());
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TESTS_HLO_LEGACY_GPU_TEST_BASE_H_
