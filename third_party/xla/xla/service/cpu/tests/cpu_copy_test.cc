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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/cpu/fusion_wrapper.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

TEST_F(HloTestBase, SubByteEqualShapeCopy) {
  const std::string hlo_text = R"hlo(
HloModule module

ENTRY entry {
  in = u2[20,20]{1,0:E(2)} iota(), iota_dimension=1
  copy = u2[20,20]{1,0:E(2)} copy(in)
  ROOT out = u8[20,20]{1,0} convert(copy)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TargetMachineFeaturesStub target_machine_features(
      [](int64_t size) { return 16; });
  FusionWrapper fusion_wrapper(&target_machine_features);
  TF_ASSERT_OK(fusion_wrapper.Run(module.get()).status());
  TF_ASSERT_OK_AND_ASSIGN(
      const Literal result,
      Execute(std::move(module), {}, /*run_hlo_passes=*/false));

  absl::Span<const uint8_t> result_data = result.data<uint8_t>();
  for (int64_t row = 0; row < 20; ++row) {
    for (int64_t col = 0; col < 20; ++col) {
      EXPECT_EQ(result_data[row * 20 + col], col % 4);
    }
  }
}

TEST_F(HloTestBase, LayoutChangingSubByteCopyFails) {
  const std::string hlo_text = R"hlo(
HloModule module

ENTRY entry {
  in = u2[20,20]{1,0:E(2)} iota(), iota_dimension=1
  transpose = u2[20,20]{0,1:E(2)} transpose(in), dimensions={1,0}
  copy = u2[20,20]{1,0:E(2)} copy(transpose)
  ROOT out = u8[20,20]{1,0} convert(copy)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TargetMachineFeaturesStub target_machine_features(
      [](int64_t size) { return 16; });
  FusionWrapper fusion_wrapper(&target_machine_features);
  TF_ASSERT_OK(fusion_wrapper.Run(module.get()).status());
  EXPECT_FALSE(Execute(std::move(module), {}, /*run_hlo_passes=*/false).ok());
}

}  // namespace
}  // namespace xla::cpu
