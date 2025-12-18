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

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

TEST_F(CpuCodegenTest, SubByteCopy) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      const Literal result,
      Execute(std::move(module), {}, /*run_hlo_passes=*/false));

  absl::Span<const uint8_t> result_data = result.data<uint8_t>();
  for (int64_t row = 0; row < 20; ++row) {
    for (int64_t col = 0; col < 20; ++col) {
      EXPECT_EQ(result_data[row * 20 + col], row % 4);
    }
  }
}

}  // namespace
}  // namespace xla::cpu
