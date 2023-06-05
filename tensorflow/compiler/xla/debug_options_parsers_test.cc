/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/debug_options_parsers.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {

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

}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
