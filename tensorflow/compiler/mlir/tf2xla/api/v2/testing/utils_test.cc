/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/testing/utils.h"

#include <stdlib.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace testing {
namespace {

class UtilsTest : public ::testing::Test {};

TEST_F(UtilsTest, TestDataPathSucceeds) {
  std::string expected_test_data_path_regex =
      ".*tensorflow/compiler/mlir/tf2xla/api/v2/testdata/";

  std::string result_test_data_path = TestDataPath();

  EXPECT_THAT(result_test_data_path,
              ::testing::ContainsRegex(expected_test_data_path_regex));
}

}  // namespace
}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
