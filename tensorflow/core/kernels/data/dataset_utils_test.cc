/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/dataset_utils.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(DatasetUtils, ComputeMoveVector) {
  struct TestCase {
    std::vector<int> indices;
    std::vector<bool> expected;
  };

  TestCase test_cases[] = {
      TestCase{{}, {}},
      TestCase{{1}, {true}},
      TestCase{{1, 1}, {false, true}},
      TestCase{{1, 2}, {true, true}},
      TestCase{{1, 1, 2}, {false, true, true}},
      TestCase{{1, 2, 2}, {true, false, true}},
  };

  for (auto& test_case : test_cases) {
    EXPECT_EQ(test_case.expected, ComputeMoveVector(test_case.indices));
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
