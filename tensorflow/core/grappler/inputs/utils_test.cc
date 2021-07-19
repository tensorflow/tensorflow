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

#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class UtilsTest : public ::testing::Test {
 protected:
  string BaseDir() { return io::JoinPath(testing::TmpDir(), "base_dir"); }

  void SetUp() override {
    TF_CHECK_OK(env_->CreateDir(BaseDir()));
    non_existent_file_ = io::JoinPath(BaseDir(), "non_existent_file.txt");
    actual_file_ = io::JoinPath(BaseDir(), "test_file.txt");
    TF_CHECK_OK(WriteStringToFile(env_, actual_file_, "Some test data"));

    text_graph_def_file_ = io::JoinPath(BaseDir(), "text_graph_def_file.txt");
    binary_graph_def_file_ =
        io::JoinPath(BaseDir(), "binary_graph_def_file.txt");
    text_meta_graph_def_file_ =
        io::JoinPath(BaseDir(), "text_meta_graph_def_file.txt");
    binary_meta_graph_def_file_ =
        io::JoinPath(BaseDir(), "binary_meta_graph_def_file.txt");

    auto node = graph_def_.add_node();
    node->set_name("foo");
    node->set_op("bar");
    TF_CHECK_OK(WriteTextProto(env_, text_graph_def_file_, graph_def_));
    TF_CHECK_OK(WriteBinaryProto(env_, binary_graph_def_file_, graph_def_));
    *meta_graph_def_.mutable_graph_def() = graph_def_;
    TF_CHECK_OK(
        WriteTextProto(env_, text_meta_graph_def_file_, meta_graph_def_));
    TF_CHECK_OK(
        WriteBinaryProto(env_, binary_meta_graph_def_file_, meta_graph_def_));
  }

  void TearDown() override {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(
        env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs));
  }

  GraphDef graph_def_;
  MetaGraphDef meta_graph_def_;
  string non_existent_file_;
  string actual_file_;
  string text_graph_def_file_;
  string binary_graph_def_file_;
  string text_meta_graph_def_file_;
  string binary_meta_graph_def_file_;
  Env* env_ = Env::Default();
};

TEST_F(UtilsTest, FilesExist) {
  EXPECT_FALSE(FilesExist(std::vector<string>{{non_existent_file_}}));
  EXPECT_FALSE(
      FilesExist(std::vector<string>{{non_existent_file_}, {actual_file_}}));
  EXPECT_TRUE(FilesExist(std::vector<string>{{actual_file_}}));

  std::vector<Status> status;
  EXPECT_FALSE(FilesExist(
      std::vector<string>{{non_existent_file_}, {actual_file_}}, &status));
  EXPECT_EQ(status.size(), 2);
  EXPECT_FALSE(status[0].ok());
  EXPECT_TRUE(status[1].ok());
}

TEST_F(UtilsTest, ReadGraphDefFromFile_Text) {
  GraphDef result;
  TF_CHECK_OK(ReadGraphDefFromFile(text_graph_def_file_, &result));
  EXPECT_EQ(result.DebugString(), graph_def_.DebugString());
}

TEST_F(UtilsTest, ReadGraphDefFromFile_Binary) {
  GraphDef result;
  TF_CHECK_OK(ReadGraphDefFromFile(binary_graph_def_file_, &result));
  EXPECT_EQ(result.DebugString(), graph_def_.DebugString());
}

TEST_F(UtilsTest, ReadMetaGraphDefFromFile_Text) {
  MetaGraphDef result;
  TF_CHECK_OK(ReadMetaGraphDefFromFile(text_meta_graph_def_file_, &result));
  EXPECT_EQ(result.DebugString(), meta_graph_def_.DebugString());
}

TEST_F(UtilsTest, ReadReadMetaGraphDefFromFile_Binary) {
  MetaGraphDef result;
  TF_CHECK_OK(ReadMetaGraphDefFromFile(binary_meta_graph_def_file_, &result));
  EXPECT_EQ(result.DebugString(), meta_graph_def_.DebugString());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
