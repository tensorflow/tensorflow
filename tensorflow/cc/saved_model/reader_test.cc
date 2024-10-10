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

#include "tensorflow/cc/saved_model/reader.h"

#include <gmock/gmock.h>
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"

namespace tensorflow {
namespace {

string TestDataPbTxt() {
  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "half_plus_two_pbtxt", "00000123");
}

string TestDataSharded() {
  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "half_plus_two", "00000123");
}

string ChunkedSavedModel() {
  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "chunked_saved_model", "chunked_model");
}

string NonChunkedSavedModel() {
  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "chunked_saved_model", "non_chunked_model");
}

class ReaderTest : public ::testing::Test {
 protected:
  ReaderTest() {}

  void CheckMetaGraphDef(const MetaGraphDef& meta_graph_def) {
    const auto& tags = meta_graph_def.meta_info_def().tags();
    EXPECT_TRUE(std::find(tags.begin(), tags.end(), kSavedModelTagServe) !=
                tags.end());
    EXPECT_NE(meta_graph_def.meta_info_def().tensorflow_version(), "");
    EXPECT_EQ(
        meta_graph_def.signature_def().at("serving_default").method_name(),
        "tensorflow/serving/predict");
  }
};

TEST_F(ReaderTest, TagMatch) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(export_dir, {kSavedModelTagServe},
                                              &meta_graph_def));
  CheckMetaGraphDef(meta_graph_def);
}

TEST_F(ReaderTest, NoTagMatch) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  absl::Status st = ReadMetaGraphDefFromSavedModel(export_dir, {"missing-tag"},
                                                   &meta_graph_def);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.message(),
      "Could not find meta graph def matching supplied tags: { missing-tag }"))
      << st.message();
}

TEST_F(ReaderTest, NoTagMatchMultiple) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  absl::Status st = ReadMetaGraphDefFromSavedModel(
      export_dir, {kSavedModelTagServe, "missing-tag"}, &meta_graph_def);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.message(), "Could not find meta graph def matching supplied tags: "))
      << st.message();
}

TEST_F(ReaderTest, InvalidExportPath) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath("missing-path");
  absl::Status st = ReadMetaGraphDefFromSavedModel(
      export_dir, {kSavedModelTagServe}, &meta_graph_def);
  EXPECT_FALSE(st.ok());
}

TEST_F(ReaderTest, ReadSavedModelDebugInfoIfPresent) {
  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  std::unique_ptr<GraphDebugInfo> debug_info_proto;
  TF_ASSERT_OK(ReadSavedModelDebugInfoIfPresent(export_dir, &debug_info_proto));
}

TEST_F(ReaderTest, MetricsNotUpdatedFailedRead) {
  MetaGraphDef meta_graph_def;
  const int read_count_v1 = metrics::SavedModelReadCount("1").value();
  const int read_count_v2 = metrics::SavedModelReadCount("2").value();

  const string export_dir = GetDataDependencyFilepath("missing-path");
  absl::Status st =
      ReadMetaGraphDefFromSavedModel(export_dir, {"serve"}, &meta_graph_def);

  EXPECT_FALSE(st.ok());
  EXPECT_EQ(metrics::SavedModelReadCount("1").value(), read_count_v1);
  EXPECT_EQ(metrics::SavedModelReadCount("2").value(), read_count_v2);
}

TEST_F(ReaderTest, MetricsUpdatedSuccessfulRead) {
  MetaGraphDef meta_graph_def;
  const int read_count_v1 = metrics::SavedModelReadCount("1").value();

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  absl::Status st =
      ReadMetaGraphDefFromSavedModel(export_dir, {"serve"}, &meta_graph_def);
  EXPECT_EQ(metrics::SavedModelReadCount("1").value(), read_count_v1 + 1);
}

// Placeholder for protosplitter merger merge test.

}  // namespace
}  // namespace tensorflow
