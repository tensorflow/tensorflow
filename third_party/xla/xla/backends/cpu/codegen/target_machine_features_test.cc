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

#include "xla/backends/cpu/codegen/target_machine_features.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"

namespace xla::cpu {
namespace {

struct Avx512Bf16TestSpec {
  std::string cpu_name;
  std::string features;
  bool has_avx512bf16;
};

class Avx512Bf16Test
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<Avx512Bf16TestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<Avx512Bf16TestSpec>& info) {
    return info.param.cpu_name;
  }
};

TEST_P(Avx512Bf16Test, CheckAvailability) {
  Avx512Bf16TestSpec spec = GetParam();
  const char* triple_string = "x86_64-unknown-linux-gnu";
  std::unique_ptr<TargetMachineFeatures> features =
      CreateTargetMachineFeatures(triple_string, spec.cpu_name, spec.features);
  EXPECT_EQ(features->has_avx512bf16(), spec.has_avx512bf16);
}

std::vector<Avx512Bf16TestSpec> GetAvx512Bf16TestSpecs() {
  return std::vector<Avx512Bf16TestSpec>{
      Avx512Bf16TestSpec{"znver3", "+avx,+avx2", false},
      Avx512Bf16TestSpec{"sapphirerapids",
                         "+avx512vnni,+avx512bf16,+amx-bf16,+amx-int8,"
                         "+amx-tile,+amx-transpose",
                         true}};
}

INSTANTIATE_TEST_SUITE_P(Avx512Bf16Suite, Avx512Bf16Test,
                         ::testing::ValuesIn(GetAvx512Bf16TestSpecs()),
                         Avx512Bf16Test::Name);

}  // namespace
}  // namespace xla::cpu
