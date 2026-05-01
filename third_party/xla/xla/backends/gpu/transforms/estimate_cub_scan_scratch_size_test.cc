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

#include "xla/backends/gpu/transforms/estimate_cub_scan_scratch_size.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class EstimateCubScanScratchSizeTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  void SetUp() override {
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>::SetUp();
    ASSERT_OK_AND_ASSIGN(test_platform_, PlatformUtil::GetPlatform("gpu"));
  }

  void RunAndCheck(absl::string_view hlo, absl::string_view expected) {
    RunAndFilecheckHloRewrite(
        hlo, EstimateCubScanScratchSize(GetTestPlatform()->Name()), expected);
  }

  const stream_executor::Platform* GetTestPlatform() const {
    return test_platform_;
  }

 private:
  stream_executor::Platform* test_platform_ = nullptr;
};

TEST_F(EstimateCubScanScratchSizeTest, BasicScan) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %input = f32[100] parameter(0)
      %custom-call = (f32[100]{0}, u8[1]{0})
        custom-call(%input),
        custom_call_target="xla.gpu.ext.cub_scan_unassigned_scratch_size",
        backend_config={"vector_length":1, "row_length":1, "column_length":100, "kind":1, "is_reverse":false}
      ROOT %t = f32[100]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_scan"
    // CHECK-SAME: api_version=API_VERSION_TYPED_FFI
  )");
}

}  // namespace
}  // namespace xla::gpu
