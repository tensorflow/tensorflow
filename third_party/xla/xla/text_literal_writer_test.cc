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

#include "xla/text_literal_writer.h"

#include <string>

#include "xla/literal_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"

namespace xla {
namespace {

TEST(TextLiteralWriterTest, WritesFloatLiteral) {
  auto literal = LiteralUtil::CreateR2<float>({
      {3.14, 2.17},
      {1.23, 4.56},
  });
  std::string path;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&path));
  ASSERT_IS_OK(TextLiteralWriter::WriteToPath(literal, path));
  std::string contents;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), path, &contents));
  const std::string expected = R"(f32[2,2]
(0, 0): 3.14
(0, 1): 2.17
(1, 0): 1.23
(1, 1): 4.56
)";
  EXPECT_EQ(expected, contents);
}

}  // namespace
}  // namespace xla
