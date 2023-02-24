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

#include "tensorflow/python/framework/python_op_gen_annotator.h"

#include <utility>

#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/framework/kythe_metadata.pb.h"

namespace tensorflow {
namespace python_op_gen_internal {
namespace {

using ::testing::StartsWith;

GeneratedCodeInfo ParseMetadata(string metadata) {
  GeneratedCodeInfo generated_code_info;
  std::pair<string, string> p = absl::StrSplit(metadata, ':');
  string serialized_generated_code_info;
  absl::Base64Unescape(p.second, &serialized_generated_code_info);
  generated_code_info.ParseFromString(serialized_generated_code_info);
  return generated_code_info;
}

TEST(PythonOpGenAnnotatorTest, AddAnnotationWithoutSourceOffsets) {
  GeneratedCodeAnnotator annotator;
  OpDef fakeOpDef;
  fakeOpDef.set_name("fake_op");
  annotator.AddAnnotation(fakeOpDef, "fake_op", 0);
  string meta = annotator.BuildKytheMetadata();
  ASSERT_THAT(meta, StartsWith("# kythe.proto.metadata.GeneratedCodeInfo:"));
  GeneratedCodeInfo actual = ParseMetadata(meta);
  GeneratedCodeInfo expected;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString("type: KYTHE0", &expected));
  EXPECT_EQ(actual.SerializeAsString(), expected.SerializeAsString());
}

TEST(PythonOpGenAnnotatorTest, AddAnnotationWithSourceOffsets) {
  GeneratedCodeAnnotator annotator;
  OpDef fakeOpDef;
  fakeOpDef.set_name("fake_op");
  OpRegOffsets fakeOffsets;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      R"pb(
        offsets {
          name: "fake_op",
          filepath: "file/path/to/fake_op.cc",
          start: 7,
          end: 11,
        }
      )pb",
      &fakeOffsets));
  annotator.AddAnnotation(fakeOpDef, "fake_op", 100);
  annotator.FillSourceOffsets(fakeOffsets);

  string meta = annotator.BuildKytheMetadata();
  ASSERT_THAT(meta, StartsWith("# kythe.proto.metadata.GeneratedCodeInfo:"));
  GeneratedCodeInfo actual = ParseMetadata(meta);

  EXPECT_EQ(actual.meta(0).type(), MappingRule::ANCHOR_ANCHOR);
  EXPECT_EQ(actual.meta(0).edge(), "/kythe/edge/imputes");
  EXPECT_EQ(actual.meta(0).source_vname().path(), "file/path/to/fake_op.cc");
  EXPECT_EQ(actual.meta(0).source_begin(), 7);
  EXPECT_EQ(actual.meta(0).source_end(), 11);
  EXPECT_EQ(actual.meta(0).target_begin(), 100);
  EXPECT_EQ(actual.meta(0).target_end(), 107);
}

TEST(PythonOpGenAnnotatorTest, AddAnnotationWithSourceOffsetsAndNonZeroBase) {
  GeneratedCodeAnnotator annotator;
  OpDef fakeOpDef;
  fakeOpDef.set_name("fake_op");
  OpRegOffsets fakeOffsets;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      R"pb(
        offsets {
          name: "fake_op",
          filepath: "file/path/to/fake_op.cc",
          start: 7,
          end: 11,
        }
      )pb",
      &fakeOffsets));
  annotator.SetBase(10);
  annotator.AddAnnotation(fakeOpDef, "fake_op", 100);
  annotator.FillSourceOffsets(fakeOffsets);

  string meta = annotator.BuildKytheMetadata();
  ASSERT_THAT(meta, StartsWith("# kythe.proto.metadata.GeneratedCodeInfo:"));
  GeneratedCodeInfo actual = ParseMetadata(meta);

  EXPECT_EQ(actual.meta(0).type(), MappingRule::ANCHOR_ANCHOR);
  EXPECT_EQ(actual.meta(0).edge(), "/kythe/edge/imputes");
  EXPECT_EQ(
      actual.meta(0).source_vname().signature(),
      absl::StrFormat("@7:11@tensorflow/op#fake_op#%s#file/path/to/fake_op.cc",
                      kKytheCorpus));
  EXPECT_EQ(actual.meta(0).source_vname().path(), "file/path/to/fake_op.cc");
  EXPECT_EQ(actual.meta(0).source_begin(), 7);
  EXPECT_EQ(actual.meta(0).source_end(), 11);
  EXPECT_EQ(actual.meta(0).target_begin(), 110);
  EXPECT_EQ(actual.meta(0).target_end(), 117);
}

TEST(PythonOpGenAnnotatorTest, AddMultipleAnnotation) {
  GeneratedCodeAnnotator annotator;
  OpDef fakeOpDef;
  OpRegOffsets fakeOffsets;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      R"pb(
        offsets {
          name: "fake_op_1",
          filepath: "file/path/to/fake_op.cc",
          start: 7,
          end: 11,
        }
        offsets {
          name: "fake_op_2",
          filepath: "file/path/to/fake_op.cc",
          start: 101,
          end: 103,
        }
      )pb",
      &fakeOffsets));
  fakeOpDef.set_name("fake_op_1");
  annotator.AddAnnotation(fakeOpDef, "fake_op_1", 10);
  fakeOpDef.set_name("fake_op_2");
  annotator.AddAnnotation(fakeOpDef, "fake_op_2", 100);
  annotator.FillSourceOffsets(fakeOffsets);

  string meta = annotator.BuildKytheMetadata();
  ASSERT_THAT(meta, StartsWith("# kythe.proto.metadata.GeneratedCodeInfo:"));
  GeneratedCodeInfo actual = ParseMetadata(meta);

  EXPECT_EQ(actual.meta_size(), 2);
}

}  // namespace
}  // namespace python_op_gen_internal
}  // namespace tensorflow
