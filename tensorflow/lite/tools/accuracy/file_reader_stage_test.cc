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

#include <cstdio>
#include <fstream>
#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/accuracy/file_reader_stage.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace metrics {
namespace {

class TempFile {
 public:
  TempFile() {
    string file_path;
    if (Env::Default()->LocalTempFilename(&file_path)) {
      file_path_ = file_path;
      created_ = true;
    }
  }

  string filepath() { return file_path_; }
  bool CreateFileWithContents(const std::string& contents) {
    if (!created_) {
      return false;
    }
    std::fstream file(file_path_, std::ios_base::out);
    if (file) {
      file << contents;
    }
    return file.good();
  }

  ~TempFile() {
    if (created_) {
      std::remove(file_path_.c_str());
    }
  }

 private:
  bool created_ = false;
  string file_path_;
};

TEST(FileReaderStageTest, FileIsRead) {
  TempFile file;
  const string kFileContents = "Hello world.";
  ASSERT_TRUE(file.CreateFileWithContents(kFileContents));
  Scope scope = Scope::NewRootScope();
  FileReaderStage reader_stage;
  reader_stage.AddToGraph(scope, file.filepath());
  TF_CHECK_OK(scope.status());
  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));
  std::vector<Tensor> outputs;
  auto run_status =
      session->Run({},                               /*inputs*/
                   {reader_stage.output_name()}, {}, /*target node names */
                   &outputs);
  TF_CHECK_OK(run_status);
  EXPECT_EQ(1, outputs.size());
  string contents = outputs[0].scalar<string>()();
  EXPECT_EQ(kFileContents, contents);
}

TEST(FileReaderStageTest, InvalidFile) {
  Scope scope = Scope::NewRootScope();
  FileReaderStage reader_stage;
  reader_stage.AddToGraph(scope, string("non_existent_file"));
  TF_CHECK_OK(scope.status());
  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));
  std::vector<Tensor> outputs;
  auto run_status =
      session->Run({},                               /*inputs*/
                   {reader_stage.output_name()}, {}, /*target node names */
                   &outputs);
  EXPECT_FALSE(run_status.ok());
}

}  // namespace

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
