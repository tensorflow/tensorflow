// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/byte_splitter.h"

#include <memory>

#include <gmock/gmock.h>

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAre;

TEST(ByteSplitterTest, SplitAscii) {
  const absl::string_view input_string("hello");
  std::vector<unsigned char> output_bytes;
  std::vector<int> output_offsets;
  ByteSplitter s;
  s.Split(input_string, &output_bytes, &output_offsets);
  EXPECT_THAT(output_bytes, ElementsAre(104, 101, 108, 108, 111));
  EXPECT_THAT(output_offsets, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(ByteSplitterTest, SplitUnicode) {
  const absl::string_view input_string("muñdʓ");
  std::vector<unsigned char> output_bytes;
  std::vector<int> output_offsets;
  ByteSplitter s;
  s.Split(input_string, &output_bytes, &output_offsets);
  EXPECT_THAT(output_bytes, ElementsAre(109, 117, 195, 177, 100, 202, 147));
  EXPECT_THAT(output_offsets, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(ByteSplitterTest, SplitEmoji) {
  const absl::string_view input_string("😀🙃");
  std::vector<unsigned char> output_bytes;
  std::vector<int> output_offsets;
  ByteSplitter s;
  s.Split(input_string, &output_bytes, &output_offsets);
  EXPECT_THAT(output_bytes,
              ElementsAre(240, 159, 152, 128, 240, 159, 153, 131));
  EXPECT_THAT(output_offsets, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
}

TEST(ByteSplitterTest, SplitHanzi) {
  const absl::string_view input_string("你好");
  std::vector<unsigned char> output_bytes;
  std::vector<int> output_offsets;
  ByteSplitter s;
  s.Split(input_string, &output_bytes, &output_offsets);
  EXPECT_THAT(output_bytes, ElementsAre(228, 189, 160, 229, 165, 189));
  EXPECT_THAT(output_offsets, ElementsAre(0, 1, 2, 3, 4, 5, 6));
}

TEST(ByteSplitterTest, SplitByBytesHanzi) {
  ByteSplitter s;
  auto output = s.SplitByOffsets("你好", {0, 3}, {3, 6});
  EXPECT_TRUE(output.ok());
  EXPECT_THAT(output.value(), ElementsAre("你", "好"));
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
