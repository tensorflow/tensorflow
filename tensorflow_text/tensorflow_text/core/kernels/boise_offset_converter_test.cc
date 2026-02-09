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

#include "tensorflow_text/core/kernels/boise_offset_converter.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

using ::testing::ContainerEq;

namespace tensorflow {
namespace text {
namespace {

// Helper function to extract texts based on the begin and end offsets.
// content = "Who let the dogs out"
// begin_offsets = {12, 17}
// end_offsets = {16, 20}
// Foo returns: {"dogs", "out"}
std::vector<std::string> ExtractTextsFromOffsets(
    const std::string content, const std::vector<int> begin_offsets,
    const std::vector<int> end_offsets) {
  absl::string_view content_sv = absl::string_view(content);
  std::vector<std::string> res;
  for (int i = 0; i < begin_offsets.size(); ++i) {
    int text_len = end_offsets[i] - begin_offsets[i];
    res.push_back(static_cast<std::string>(
        content_sv.substr(begin_offsets[i], text_len)));
  }
  return res;
}

// Test that we can transform offsets into BOISE tags
TEST(OffsetsToBoiseTagsTest, ExtractSingleton) {
  //                               1         2
  //                     012345678901234567890
  std::string content = "Who let the dogs out";
  std::string entity = "dogs";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 17};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<int> entity_begin_offsets = {12};
  std::vector<int> entity_end_offsets = {16};
  std::vector<std::string> entity_type = {"animal"};
  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "O", "O", "O", "S-animal", "O"}));
}

TEST(OffsetsToBoiseTagsTest, ExtractSingletonStrictBoundary) {
  //                               1
  //                     01234567890123456789
  std::string content = "Who let the dogs out";
  std::string entity = "dogs";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 17};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<int> entity_begin_offsets = {13};
  std::vector<int> entity_end_offsets = {16};
  std::vector<std::string> entity_type = {"animal"};
  bool use_strict_boundary_mode = true;
  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type,
                         use_strict_boundary_mode)
          .ValueOrDie();
  EXPECT_THAT(boise_tags,
              ContainerEq(std::vector<std::string>{"O", "O", "O", "O", "O"}));
}

TEST(OffsetsToBoiseTagsTest, ExtractBEEntity) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::string entity = "german shepherd";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<int> entity_begin_offsets = {12};
  std::vector<int> entity_end_offsets = {27};
  std::vector<std::string> entity_type = {"animal"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "O", "O", "O", "B-animal", "E-animal", "O"}));
}

TEST(OffsetsToBoiseTagsTest, ExtractBIEEntity) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "How big is Los Angeles County?";
  std::string entity = "Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 4, 8, 11, 15, 23, 29};
  std::vector<int> token_end_offsets = {3, 7, 10, 14, 22, 29, 30};
  std::vector<int> entity_begin_offsets = {11};
  std::vector<int> entity_end_offsets = {29};
  std::vector<std::string> entity_type = {"loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "O", "O", "O", "B-loc", "I-loc", "E-loc", "O"}));
}

TEST(OffsetsToBoiseTagsTest, ExtractMutipleEntities) {
  //                               1         2         3
  //                     01234567890123456789012345678901234567
  std::string content = "Getty Center is in Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 6, 13, 16, 19, 23, 31};
  std::vector<int> token_end_offsets = {5, 12, 15, 18, 22, 30, 37};
  std::vector<int> entity_begin_offsets = {0, 19};
  std::vector<int> entity_end_offsets = {12, 37};
  std::vector<std::string> entity_type = {"org", "loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags,
              ContainerEq(std::vector<std::string>{"B-org", "E-org", "O", "O",
                                                   "B-loc", "I-loc", "E-loc"}));
}

TEST(OffsetsToBoiseTagsTest, LooseBoundary) {
  //                               1         2         3
  //                     01234567890123456789012345678901234567
  std::string content = "Getty Center is in Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 6, 13, 16, 19, 23, 31};
  std::vector<int> token_end_offsets = {5, 12, 15, 18, 22, 30, 37};
  std::vector<int> entity_begin_offsets = {3, 19};
  std::vector<int> entity_end_offsets = {10, 32};
  std::vector<std::string> entity_type = {"org", "loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags,
              ContainerEq(std::vector<std::string>{"B-org", "E-org", "O", "O",
                                                   "B-loc", "I-loc", "E-loc"}));
}

TEST(OffsetsToBoiseTagsTest, StrictBoundary) {
  //                               1         2         3
  //                     01234567890123456789012345678901234567
  std::string content = "Getty Center is in Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 6, 13, 16, 19, 23, 31};
  std::vector<int> token_end_offsets = {5, 12, 15, 18, 22, 30, 37};
  std::vector<int> entity_begin_offsets = {3, 19};
  std::vector<int> entity_end_offsets = {12, 32};
  std::vector<std::string> entity_type = {"org", "loc"};
  bool use_strict_boundary_mode = true;

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type,
                         use_strict_boundary_mode)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "O", "E-org", "O", "O", "B-loc", "I-loc", "O"}));
}

TEST(OffsetsToBoiseTagsTest, OneTokenMultiEntitiesLastPrecedes) {
  //                               1
  //                     0123456789012
  std::string content = "Getty Center";
  std::vector<int> token_begin_offsets = {0};
  std::vector<int> token_end_offsets = {12};
  std::vector<int> entity_begin_offsets = {0, 6};
  std::vector<int> entity_end_offsets = {5, 12};
  std::vector<std::string> entity_type = {"per", "loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{"B-loc"}));
}

TEST(OffsetsToBoiseTagsTest, OneTokenMultEntitiesPartialOverlapLastPrecedes) {
  //                               1
  //                     0123456789012
  std::string content = "Getty Center";
  std::vector<int> token_begin_offsets = {0, 6};
  std::vector<int> token_end_offsets = {5, 12};
  std::vector<int> entity_begin_offsets = {0, 9};
  std::vector<int> entity_end_offsets = {8, 12};
  std::vector<std::string> entity_type = {"per", "loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags,
              ContainerEq(std::vector<std::string>{"B-per", "B-loc"}));
}

TEST(OffsetsToBoiseTagsTest, MultiTokensOneEntityPartialOverlapLastPrecedes) {
  //                               1         2         3
  //                     01234567890123456789012345678901234
  std::string content = "Getty Center, Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 6, 14, 18, 26};
  std::vector<int> token_end_offsets = {5, 12, 17, 25, 32};
  std::vector<int> entity_begin_offsets = {0, 15};
  std::vector<int> entity_end_offsets = {14, 30};
  std::vector<std::string> entity_type = {"org", "loc"};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "B-org", "I-org", "B-loc", "I-loc", "E-loc"}));
}

TEST(OffsetsToBoiseTagsTest, EmptySpanOffsets) {
  std::vector<int> token_begin_offsets = {0, 6, 13, 16, 19, 23, 31};
  std::vector<int> token_end_offsets = {5, 12, 15, 18, 22, 30, 37};
  std::vector<int> entity_begin_offsets = {};
  std::vector<int> entity_end_offsets = {};
  std::vector<std::string> entity_type = {};

  std::vector<std::string> boise_tags =
      OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                         entity_begin_offsets, entity_end_offsets, entity_type)
          .ValueOrDie();
  EXPECT_THAT(boise_tags, ContainerEq(std::vector<std::string>{
                              "O", "O", "O", "O", "O", "O", "O"}));
}

TEST(OffsetsToBoiseTagsTest, InputSizeError) {
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 17};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<int> entity_begin_offsets = {12};
  std::vector<int> entity_end_offsets = {16};
  std::vector<std::string> entity_type = {"animal", "extra_entity"};
  EXPECT_FALSE(OffsetsToBoiseTags(token_begin_offsets, token_end_offsets,
                                  entity_begin_offsets, entity_end_offsets,
                                  entity_type)
                   .ok());
}

// Test that BOISE tags can be transformed into offets
TEST(BoiseTagsToOffsetTest, BeginAndEndTagsAreConvertedToOffsets) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",        "O",        "O",
                                         "B-animal", "E-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, SingletonTagsAreExtracted) {
  //                               1         2
  //                     012345678901234567890
  std::string content = "Who let the dogs out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 17};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<std::string> boise_tags = {"O", "O", "O", "S-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"dogs"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, BeginInsideAndEndLabelsAreExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "How big is Los Angeles County?";
  std::vector<int> token_begin_offsets = {0, 4, 8, 11, 15, 23, 29};
  std::vector<int> token_end_offsets = {3, 7, 10, 14, 22, 29, 30};
  std::vector<std::string> boise_tags = {"O",     "O",     "O", "B-loc",
                                         "I-loc", "E-loc", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts,
              ContainerEq(std::vector<std::string>{"Los Angeles County"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"loc"}));
}

TEST(BoiseTagsToOffsetTest, InsideEndLabelsAreExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",        "O",        "O",
                                         "I-animal", "E-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, BeginInsideLabelsAreExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",        "O",        "O",
                                         "B-animal", "I-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, InsideOnlyLabelIsExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 21};
  std::vector<int> token_end_offsets = {3, 7, 11, 20, 24};
  std::vector<std::string> boise_tags = {
      "O", "O", "O", "I-animal", "O",
  };

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, BeginOnlyLabelIsExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 21};
  std::vector<int> token_end_offsets = {3, 7, 11, 20, 24};
  std::vector<std::string> boise_tags = {
      "O", "O", "O", "B-animal", "O",
  };

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, EndOnlyLabelIsExtracted) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 21};
  std::vector<int> token_end_offsets = {3, 7, 11, 20, 24};
  std::vector<std::string> boise_tags = {
      "O", "O", "O", "E-animal", "O",
  };

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, MultipleEntitiesAreExtracted) {
  //                               1         2         3
  //                     01234567890123456789012345678901234567
  std::string content = "Getty Center is in Los Angeles County";
  std::vector<int> token_begin_offsets = {0, 6, 13, 16, 19, 23, 31};
  std::vector<int> token_end_offsets = {5, 12, 15, 18, 22, 30, 37};
  std::vector<std::string> boise_tags = {"B-org", "E-org", "O",    "O",
                                         "B-loc", "I-loc", "E-loc"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{
                         "Getty Center", "Los Angeles County"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"org", "loc"}));
}

TEST(BoiseTagsToOffsetTest, MultipleBeginLabels) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",     "O",        "O",
                                         "B-loc", "B-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, MultipleInsideLabels) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",     "O",        "O",
                                         "I-loc", "I-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, MultipleEndLabels) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",     "O",        "O",
                                         "E-loc", "E-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts,
              ContainerEq(std::vector<std::string>{"german", "shepherd"}));
  EXPECT_THAT(span_types,
              ContainerEq(std::vector<std::string>{"loc", "animal"}));
}

TEST(BoiseTagsToOffsetTest, MultipleSingleLabels) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who let the german shepherd out";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19, 28};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27, 31};
  std::vector<std::string> boise_tags = {"O",     "O",        "O",
                                         "S-loc", "S-animal", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts,
              ContainerEq(std::vector<std::string>{"german", "shepherd"}));
  EXPECT_THAT(span_types,
              ContainerEq(std::vector<std::string>{"loc", "animal"}));
}

TEST(BoiseTagsToOffsetTest, TrailingBeginLabels) {
  //                               1         2         3
  //                     0123456789012345678901234567890
  std::string content = "Who own the german shepherd";
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19};
  std::vector<int> token_end_offsets = {3, 7, 11, 18, 27};
  std::vector<std::string> boise_tags = {"O", "O", "O", "B-loc", "B-animal"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  auto texts = ExtractTextsFromOffsets(content, begin_offsets, end_offsets);
  EXPECT_THAT(texts, ContainerEq(std::vector<std::string>{"german shepherd"}));
  EXPECT_THAT(span_types, ContainerEq(std::vector<std::string>{"animal"}));
}

TEST(BoiseTagsToOffsetTest, NoBoiseLabels) {
  std::vector<int> token_begin_offsets = {0, 4, 8, 12, 19};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<std::string> boise_tags = {"O", "O", "O", "O", "O"};

  auto [begin_offsets, end_offsets, span_types] =
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ValueOrDie();

  EXPECT_TRUE(begin_offsets.empty());
  EXPECT_TRUE(end_offsets.empty());
  EXPECT_TRUE(span_types.empty());
}

TEST(BoiseTagsToOffsetTest, InputSizeError) {
  std::vector<int> token_begin_offsets = {0, 4, 8, 12};
  std::vector<int> token_end_offsets = {3, 7, 11, 16, 20};
  std::vector<std::string> boise_tags = {"O", "O", "O", "B-loc", "B-animal"};
  EXPECT_FALSE(
      BoiseTagsToOffsets(token_begin_offsets, token_end_offsets, boise_tags)
          .ok());
}

TEST(GetAllBoiseTagsFromSpanTypeTest, GetAllTagsCorrect) {
  std::vector<std::string> span_type = {"loc", "O", "per", ""};
  std::unordered_set<std::string> all_tags =
      GetAllBoiseTagsFromSpanType(span_type);
  EXPECT_THAT(all_tags, ContainerEq(std::unordered_set<std::string>{
                            "O", "B-loc", "I-loc", "S-loc", "E-loc", "B-per",
                            "I-per", "S-per", "E-per"}));
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
