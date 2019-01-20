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

#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace graph_transforms {

class FileUtilsTest : public ::testing::Test {
 protected:
  void TestLoadTextOrBinaryGraphFile() {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    const int width = 10;

    auto root = tensorflow::Scope::NewRootScope();
    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    const string text_file =
        io::JoinPath(testing::TmpDir(), "text_graph.pbtxt");
    TF_ASSERT_OK(WriteTextProto(Env::Default(), text_file, graph_def));

    const string binary_file =
        io::JoinPath(testing::TmpDir(), "binary_graph.pb");
    TF_ASSERT_OK(WriteBinaryProto(Env::Default(), binary_file, graph_def));

    const string bogus_file = io::JoinPath(testing::TmpDir(), "bogus_graph.pb");
    TF_ASSERT_OK(
        WriteStringToFile(Env::Default(), bogus_file, "Not a !{ proto..."));

    GraphDef text_graph_def;
    TF_EXPECT_OK(LoadTextOrBinaryGraphFile(text_file, &text_graph_def));
    string text_diff;
    EXPECT_TRUE(EqualGraphDef(text_graph_def, graph_def, &text_diff))
        << text_diff;

    GraphDef binary_graph_def;
    TF_EXPECT_OK(LoadTextOrBinaryGraphFile(binary_file, &binary_graph_def));
    string binary_diff;
    EXPECT_TRUE(EqualGraphDef(binary_graph_def, graph_def, &binary_diff))
        << binary_diff;

    GraphDef no_graph_def;
    EXPECT_FALSE(
        LoadTextOrBinaryGraphFile("____non_existent_file_____", &no_graph_def)
            .ok());

    GraphDef bogus_graph_def;
    EXPECT_FALSE(LoadTextOrBinaryGraphFile(bogus_file, &bogus_graph_def).ok());
  }
};

TEST_F(FileUtilsTest, TestLoadTextOrBinaryGraphFile) {
  TestLoadTextOrBinaryGraphFile();
}

}  // namespace graph_transforms
}  // namespace tensorflow
