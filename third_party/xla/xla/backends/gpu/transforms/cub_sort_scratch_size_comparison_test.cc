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
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/deviceless_estimate_cub_sort_scratch_size.h"
#include "xla/backends/gpu/transforms/estimate_cub_sort_scratch_size.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

struct TestParams {
  int key_type_size;
  int value_type_size;  // 0 means keys-only sort
  int batch_size;
  int num_items;
};

absl::StatusOr<int64_t> ExtractScratchSize(const HloModule& module);

class CubSortScratchSizeComparisonTest
    : public HloPjRtTestBase,
      public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {
 protected:
  void SetUp() override {
    HloPjRtTestBase::SetUp();
    ASSERT_OK_AND_ASSIGN(test_platform_, PlatformUtil::GetPlatform("gpu"));
  }

  absl::StatusOr<int64_t> RunPassAndExtractScratchSize(const HloModule& module,
                                                       HloModulePass* pass) {
    auto cloned_module = module.Clone();
    ASSIGN_OR_RETURN(bool changed, RunHloPass(pass, cloned_module.get()));
    if (!changed) {
      return absl::InternalError("Pass did not change the module");
    }
    return ExtractScratchSize(*cloned_module);
  }

  stream_executor::Platform* GetTestPlatform() const { return test_platform_; }

 private:
  stream_executor::Platform* test_platform_ = nullptr;
};

absl::StatusOr<absl::string_view> PrimitiveTypeToString(int bit_size) {
  switch (bit_size) {
    case 8:
      return "u8";
    case 16:
      return "u16";
    case 32:
      return "u32";
    case 64:
      return "u64";
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported bit size: %d", bit_size));
  }
}

absl::StatusOr<std::string> GetUnassignedScratchSizeSortHlo(
    const TestParams& params) {
  ASSIGN_OR_RETURN(absl::string_view key_type,
                   PrimitiveTypeToString(params.key_type_size));

  std::string dims = absl::StrFormat("[%d]", params.num_items);
  if (params.batch_size > 1) {
    dims = absl::StrFormat("[%d,%d]", params.batch_size,
                           params.num_items / params.batch_size);
  }
  std::string layout = params.batch_size > 1 ? "{1,0}" : "{0}";

  if (params.value_type_size > 0) {
    ASSIGN_OR_RETURN(absl::string_view value_type,
                     PrimitiveTypeToString(params.value_type_size));
    constexpr absl::string_view key_value_sort_hlo = R"hlo(
      HloModule m
      ENTRY main {
        %keys = $KEY_TYPE$$DIMS$ parameter(0)
        %values = $VALUE_TYPE$$DIMS$ parameter(1)
        %custom-call = ($KEY_TYPE$$DIMS$$LAYOUT$, $VALUE_TYPE$$DIMS$$LAYOUT$, u8[1]{0})
          custom-call(%keys, %values),
          custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
          backend_config={"descending":false}
        ROOT %t = $KEY_TYPE$$DIMS$$LAYOUT$ get-tuple-element(%custom-call), index=0
      }
    )hlo";
    return absl::StrReplaceAll(key_value_sort_hlo,
                               {{"$KEY_TYPE$", key_type},
                                {"$VALUE_TYPE$", value_type},
                                {"$DIMS$", dims},
                                {"$LAYOUT$", layout}});
  }

  constexpr absl::string_view keys_only_sort_hlo = R"hlo(
    HloModule m
    ENTRY main {
      %keys = $KEY_TYPE$$DIMS$ parameter(0)
      %custom-call = ($KEY_TYPE$$DIMS$$LAYOUT$, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = $KEY_TYPE$$DIMS$$LAYOUT$ get-tuple-element(%custom-call), index=0
    }
  )hlo";
  return absl::StrReplaceAll(
      keys_only_sort_hlo,
      {{"$KEY_TYPE$", key_type}, {"$DIMS$", dims}, {"$LAYOUT$", layout}});
}

absl::StatusOr<int64_t> ExtractScratchSize(const HloModule& module) {
  const HloInstruction* custom_call = nullptr;
  for (auto* comp : module.computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kCustomCall) {
        custom_call = inst;
        break;
      }
    }
    if (custom_call) {
      break;
    }
  }

  TF_RET_CHECK(custom_call != nullptr) << "Custom call not found";
  const Shape& shape = custom_call->shape();
  TF_RET_CHECK(shape.IsTuple()) << "Custom call output is not a tuple";
  const Shape& scratch_shape = shape.tuple_shapes().back();
  TF_RET_CHECK(scratch_shape.element_type() == U8)
      << "Last element of tuple is not U8";
  return scratch_shape.dimensions(0);
}

TEST_P(CubSortScratchSizeComparisonTest, CompareScratchSize) {
#ifndef NDEBUG
  GTEST_SKIP()
      << "Skipping test: Debug builds have smaller cub sort scratch space, and "
         "may break the test";
#endif
  auto [key_size, value_size, batch_size, num_items] = GetParam();
  TestParams params{key_size, value_size, batch_size, num_items};

  ASSERT_OK_AND_ASSIGN(std::string sort_hlo,
                       GetUnassignedScratchSizeSortHlo(params));
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(sort_hlo));

  // Run regular pass
  EstimateCubSortScratchSize device_pass(GetTestPlatform()->Name());
  ASSERT_OK_AND_ASSIGN(int64_t required_scratch_size,
                       RunPassAndExtractScratchSize(*module, &device_pass));

  // Run deviceless pass
  ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor,
                       GetTestPlatform()->ExecutorForDevice(0));
  const stream_executor::DeviceDescription& device_desc =
      executor->GetDeviceDescription();
  DevicelessEstimateCubSortScratchSize deviceless_pass(
      GetTestPlatform()->Name(), device_desc.name(), device_desc.cub_version());
  ASSERT_OK_AND_ASSIGN(int64_t deviceless_scratch_size,
                       RunPassAndExtractScratchSize(*module, &deviceless_pass));

  EXPECT_GE(deviceless_scratch_size, required_scratch_size)
      << "Actual scratch size is larger than the deviceless scratch size for "
         "key size "
      << key_size << ", value size " << value_size << ", batch size "
      << batch_size << ", and num items " << num_items;
  EXPECT_LE(deviceless_scratch_size, required_scratch_size * 1.1)
      << "Estimated deviceless scratch size is too large: actual scratch "
         "size needed: "
      << required_scratch_size
      << ", estimated deviceless scratch size: " << deviceless_scratch_size;
}

INSTANTIATE_TEST_SUITE_P(
    CubSortScratchSizeComparisonTestSuite, CubSortScratchSizeComparisonTest,
    ::testing::Combine(
        // key_type_size
        ::testing::Values(8, 16, 32, 64),
        // value_type_size (8 bit values aren't supported in XLA)
        ::testing::Values(0, 16, 32, 64),
        // batch_size
        ::testing::Values(1, 2, 4),
        // num_items
        ::testing::ValuesIn({
            10,
            100,
            1'000,
            10'000,
            100'000,
            1'000'000,
            2'000'000,
            4'000'000,
            10'000'000,
            100'000'000,
        })));

}  // namespace
}  // namespace xla::gpu
