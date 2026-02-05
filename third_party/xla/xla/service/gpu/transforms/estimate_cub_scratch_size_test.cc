/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/estimate_cub_scratch_size.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class EstimateCubScratchSizeTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  void SetUp() override {
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>::SetUp();
    ASSERT_OK_AND_ASSIGN(test_platform_, PlatformUtil::GetPlatform("gpu"));
  }

  void RunAndCheck(absl::string_view hlo, absl::string_view expected) {
    RunAndFilecheckHloRewrite(
        hlo, EstimateCubScratchSize(GetTestPlatform()->Name()), expected);
  }

  const stream_executor::Platform* GetTestPlatform() const {
    return test_platform_;
  }

 private:
  stream_executor::Platform* test_platform_ = nullptr;
};

// Basic sort: ascending.
TEST_F(EstimateCubScratchSizeTest, U32_F32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[1000] parameter(0)
      %values = f32[1000] parameter(1)
      %custom-call = (u32[1000]{0}, f32[1000]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = u32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u32[1000]{0}, f32[1000]{0}, u8[1]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, F32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (f32[1000]{0}, u8[1]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, S32_S32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = s32[1000] parameter(0)
      %values = s32[1000] parameter(1)
      %custom-call = (s32[1000]{0}, s32[1000]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = s32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (s32[1000]{0}, s32[1000]{0}, u8[1]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, F32_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":true}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (f32[1000]{0}, u8[1]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":true}
  )");
}

TEST_F(EstimateCubScratchSizeTest, F32_Rank3) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[10,10,10] parameter(0)
      %custom-call = (f32[10,10,10]{2,1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = f32[10,10,10]{2,1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (f32[10,10,10]{2,1,0}, u8[4756]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, F32_Rank2) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[10,100] parameter(0)
      %custom-call = (f32[10,100]{1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = f32[10,100]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (f32[10,100]{1,0}, u8[4396]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U16_F16_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = f16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":true}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[8516]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":true}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U32_F32_Rank2) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[16,128] parameter(0)
      %values = f32[16,128] parameter(1)
      %custom-call = (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = u32[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[16708]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U64_F64_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u64[16,128] parameter(0)
      %values = f64[16,128] parameter(1)
      %custom-call = (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":true}
      ROOT %t = u64[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[33092]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":true}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U16_BF16) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = bf16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[8516]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U16_BF16_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = bf16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":true}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[8516]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":true}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U16_F16) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = f16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[8516]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U32_F32_Rank2_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[16,128] parameter(0)
      %values = f32[16,128] parameter(1)
      %custom-call = (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":true}
      ROOT %t = u32[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[16708]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":true}
  )");
}

TEST_F(EstimateCubScratchSizeTest, U64_F64) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u64[16,128] parameter(0)
      %values = f64[16,128] parameter(1)
      %custom-call = (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="__cub$DeviceRadixSortUnassignedScratchSize",
        backend_config={"descending":false}
      ROOT %t = u64[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[33092]{0}) custom-call
    // CHECK-SAME: custom_call_target="__cub$DeviceRadixSort",
    // CHECK-SAME: backend_config={"descending":false}
  )");
}

}  // namespace
}  // namespace xla::gpu
