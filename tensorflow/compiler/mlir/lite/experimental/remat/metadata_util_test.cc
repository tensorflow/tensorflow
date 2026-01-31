/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h"

#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

class MetadataSerializerTest : public ::testing::Test {
 protected:
  static constexpr auto kHuge = std::numeric_limits<int32_t>::max();
  static constexpr auto kTiny = std::numeric_limits<int32_t>::min();

  std::string RoundTrip(const ModelControlDependencies &in) const {
    ModelControlDependencies out = {{{-1, -1}}};
    const std::string serialized =
        tflite::SerializeModelControlDependencies(in);
    return tflite::ParseModelControlDependencies(serialized.data(),
                                                 serialized.size(), &out)
               ? (out == in) ? "ok" : "mismatch"
               : "malformed";
  }
};

TEST_F(MetadataSerializerTest, nothing) { EXPECT_THAT(RoundTrip({}), "ok"); }

TEST_F(MetadataSerializerTest, something) {
  EXPECT_THAT(
      RoundTrip({{{1, 2}, {2, 3}, {4, 5}},
                 {},
                 {{kHuge, kTiny}, {kTiny, kHuge}, {kHuge - 1, kTiny + 1}},
                 {{1, 0}}}),
      "ok");
}

}  // namespace
}  // namespace tflite
