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

#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {
namespace {

using ::testing::Eq;

TEST(HashToolsTest, CombineHashCommutative) {
  size_t a = 0;
  size_t b = 999;

  size_t c = a;
  CombineHashCommutative(b, &c);

  size_t d = b;
  CombineHashCommutative(a, &d);

  EXPECT_THAT(c, Eq(d));
}

}  // namespace
}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
