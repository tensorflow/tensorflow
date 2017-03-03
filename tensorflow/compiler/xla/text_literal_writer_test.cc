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

#include "tensorflow/compiler/xla/text_literal_writer.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(TextLiteralWriterTest, WritesFloatLiteral) {
  auto literal = LiteralUtil::CreateR2<float>({
      {3.14, 2.17}, {1.23, 4.56},
  });
  string path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "/whatever");
  ASSERT_IS_OK(TextLiteralWriter::WriteToPath(*literal, path));
  string contents;
  TF_CHECK_OK(tensorflow::ReadFileToString(tensorflow::Env::Default(), path,
                                           &contents));
  const string expected = R"(f32[2,2]
(0, 0): 3.14
(0, 1): 2.17
(1, 0): 1.23
(1, 1): 4.56
)";
  EXPECT_EQ(expected, contents);
}

}  // namespace
}  // namespace xla
