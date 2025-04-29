/* Copyright 2017 The OpenXLA Authors.

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

// Test for parse_flags_from_env.cc

#include "xla/debug_options_parsers.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/parse_flags_from_env.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

// Test that the xla_backend_extra_options flag is parsed correctly.
TEST(DebugOptionsFlags, ParseXlaBackendExtraOptions) {
  absl::flat_hash_map<std::string, std::string> test_map;
  std::string test_string = "aa=bb,cc,dd=,ee=ff=gg";
  parse_xla_backend_extra_options(&test_map, test_string);
  EXPECT_EQ(test_map.size(), 4);
  EXPECT_EQ(test_map.at("aa"), "bb");
  EXPECT_EQ(test_map.at("cc"), "");
  EXPECT_EQ(test_map.at("dd"), "");
  EXPECT_EQ(test_map.at("ee"), "ff=gg");
}

struct UppercaseStringSetterTestSpec {
  std::string user_max_isa;
  std::string expected_max_isa;
};

class UppercaseStringSetterTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<UppercaseStringSetterTestSpec> {
 public:
  UppercaseStringSetterTest()
      : flag_values_(DefaultDebugOptionsIgnoringFlags()) {
    MakeDebugOptionsFlags(&flag_objects_, &flag_values_);
  }
  static std::string Name(
      const ::testing::TestParamInfo<UppercaseStringSetterTestSpec>& info) {
    return info.param.user_max_isa;
  }
  DebugOptions flag_values() const { return flag_values_; }
  std::vector<tsl::Flag> flag_objects() { return flag_objects_; }

 private:
  DebugOptions flag_values_;
  std::vector<tsl::Flag> flag_objects_;
};

TEST_P(UppercaseStringSetterTest, XlaCpuMaxIsa) {
  UppercaseStringSetterTestSpec spec = GetParam();
  tsl::setenv("XLA_FLAGS",
              absl::StrCat("--xla_cpu_max_isa=", spec.user_max_isa).c_str(),
              /*overwrite=*/true);

  // Parse flags from the environment variable.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects());
  EXPECT_EQ(flag_values().xla_cpu_max_isa(), spec.expected_max_isa);
}

std::vector<UppercaseStringSetterTestSpec> GetUppercaseStringSetterTestCases() {
  return std::vector<UppercaseStringSetterTestSpec>({
      UppercaseStringSetterTestSpec{"sse4_2", "SSE4_2"},
      UppercaseStringSetterTestSpec{"aVx512", "AVX512"},
      UppercaseStringSetterTestSpec{"AMx_fP16", "AMX_FP16"},
  });
}

INSTANTIATE_TEST_SUITE_P(
    UppercaseStringSetterTestInstantiation, UppercaseStringSetterTest,
    ::testing::ValuesIn(GetUppercaseStringSetterTestCases()),
    UppercaseStringSetterTest::Name);

TEST(FuelTest, FuelPassCountsAreSeparate) {
  tsl::setenv("XLA_FLAGS", "--xla_fuel=ABC=1,PQR=2", /*overwrite=*/true);
  // Parse flags from the environment variable.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  ParseDebugOptionFlagsFromEnv();

  EXPECT_TRUE(ConsumeFuel("ABC"));
  EXPECT_FALSE(ConsumeFuel("ABC"));

  EXPECT_TRUE(ConsumeFuel("PQR"));
  EXPECT_TRUE(ConsumeFuel("PQR"));
  EXPECT_FALSE(ConsumeFuel("PQR"));
}

TEST(FuelTest,
     PassFuelIsSetReturnsTrueOnExplicitlyFueledPassesAndFalseOtherwise) {
  tsl::setenv("XLA_FLAGS", "--xla_fuel=MNO=1,XYZ=2", /*overwrite=*/true);
  // Parse flags from the environment variable.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  ParseDebugOptionFlagsFromEnv();
  EXPECT_FALSE(PassFuelIsSet("ABC"));
  EXPECT_TRUE(PassFuelIsSet("MNO"));
  EXPECT_FALSE(PassFuelIsSet("PQR"));
  EXPECT_TRUE(PassFuelIsSet("XYZ"));
}
}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
