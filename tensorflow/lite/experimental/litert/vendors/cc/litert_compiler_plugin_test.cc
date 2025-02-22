// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility types for mapping LiteRt IR to arbitrary backend specific
// types. Implementations of these types define mapping for ops and tensors
// that may be used in a stndalone fashion. They also may be composed
// to create lowerings of entire graphs with topology.

#include "tensorflow/lite/experimental/litert/vendors/cc/litert_compiler_plugin.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

struct LiteRtCompilerPluginT {
  using Flag = std::pair<std::string, std::string>;
  std::vector<Flag> flags;
};

LiteRtStatus LiteRtCompilerPluginSetFlags(LiteRtCompilerPlugin compiler_plugin,
                                          LiteRtParamIndex num_flags,
                                          const char** keys,
                                          const char** values) {
  auto& flags = compiler_plugin->flags;
  flags.resize(num_flags);
  for (int i = 0; i < num_flags; ++i) {
    auto& flag = flags[i];
    flag.first = std::string(keys[i]);
    flag.second = std::string(values[i]);
  }
  return kLiteRtStatusOk;
}

namespace litert {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

TEST(CompilerFlagsTest, SetPluginFlags) {
  static constexpr const char* kKey1 = "key1";
  static constexpr const char* kKey2 = "key2";
  static constexpr const char* kKey3 = "key3";
  static constexpr const char* kValue1 = "value1";
  static constexpr const char* kEmtpyVal = "";

  LiteRtCompilerPluginT plugin;
  CompilerFlags flags;
  flags.Push(kKey1, kValue1);
  flags.Push(kKey2, kEmtpyVal);
  flags.Push(kKey3);
  LITERT_ASSERT_OK(flags.SetPluginFlags(&plugin, LiteRtCompilerPluginSetFlags));

  EXPECT_THAT(plugin.flags,
              ElementsAre(Pair(kKey1, kValue1), Pair(kKey2, kEmtpyVal),
                          Pair(kKey3, kEmtpyVal)));
}

}  // namespace
}  // namespace litert
