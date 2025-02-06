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

#include "xla/text_literal_reader.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"

namespace xla {
namespace {

TEST(TextLiteralReaderTest, ReadsR3File) {
  std::string contents = R"(f32[1,2,3]
(0,0,0): 42.5
(0,0,1): 43.5
(0,0,2): 44.5
(0,1,0): 45.5
(0,1,1): 46.5
(0,1,2): 47.5
)";

  std::string fname = tsl::testing::TmpDir() + "/ReadsR3File.data.txt";
  EXPECT_TRUE(
      tsl::WriteStringToFile(tsl::Env::Default(), fname, contents).ok());

  Literal literal = TextLiteralReader::ReadPath(fname).value();
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {1, 2, 3}), literal.shape()));
  EXPECT_EQ(42.5, literal.Get<float>({0, 0, 0}));
  EXPECT_EQ(43.5, literal.Get<float>({0, 0, 1}));
  EXPECT_EQ(44.5, literal.Get<float>({0, 0, 2}));
  EXPECT_EQ(45.5, literal.Get<float>({0, 1, 0}));
  EXPECT_EQ(46.5, literal.Get<float>({0, 1, 1}));
  EXPECT_EQ(47.5, literal.Get<float>({0, 1, 2}));
}

}  // namespace
}  // namespace xla
