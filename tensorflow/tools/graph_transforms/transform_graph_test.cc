/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/transform_graph.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to expose it in the public header.
Status ShouldIgnoreErrors(const TransformFuncParameters& transform_params,
                          bool* ignore_errors);

namespace {
Status test_empty_graph_transform(const GraphDef& graph_def,
                                  const TransformFuncContext& context,
                                  GraphDef* result) {
  result->Clear();
  return Status::OK();
}
}  // namespace

REGISTER_GRAPH_TRANSFORM("test_empty_graph_transform",
                         test_empty_graph_transform);

class TransformGraphTest : public ::testing::Test {
 protected:
  void TestConstantFolding() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const =
        Const(root.WithOpName("a_expect_removed"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const =
        Const(root.WithOpName("b_expect_removed"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add_expect_removed"), a_const, b_const);

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);

    Output mul =
        Mul(root.WithOpName("output_expect_remains"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    string graph_def_serialized;
    graph_def.SerializeToString(&graph_def_serialized);
    const string dir = testing::TmpDir();
    const string in_filename_pb = io::JoinPath(dir, "in_graphdef.pb");
    const string out_filename_pb = io::JoinPath(dir, "out_graphdef.pb");
    TF_ASSERT_OK(WriteStringToFile(Env::Default(), in_filename_pb,
                                   graph_def_serialized));

    std::vector<string> args = {"some_binary",
                                "--in_graph=" + in_filename_pb,
                                "--out_graph=" + out_filename_pb,
                                "--inputs=placeholder_expect_remains",
                                "--outputs=output_expect_remains",
                                "--transforms=fold_constants"};
    const int argc = 6;
    EXPECT_EQ(argc, args.size());
    char* argv[argc];
    std::vector<char*> char_strings;
    for (int i = 0; i < argc; ++i) {
      string arg = args[i];
      char* char_string = new char[arg.size() + 1];
      std::copy_n(arg.c_str(), arg.size() + 1, char_string);
      argv[i] = char_string;
      char_strings.push_back(char_string);
    }
    ParseFlagsAndTransformGraph(argc, argv, false);
    for (char* char_string : char_strings) {
      delete[] char_string;
    }

    GraphDef out_graph_def;
    TF_EXPECT_OK(
        ReadBinaryProto(Env::Default(), out_filename_pb, &out_graph_def));

    std::map<string, const NodeDef*> out_node_map;
    graph_transforms::MapNamesToNodes(out_graph_def, &out_node_map);

    for (const NodeDef& node : out_graph_def.node()) {
      const int occurrence_count = out_node_map.count(node.name());
      if (str_util::EndsWith(node.name(), "expect_removed")) {
        EXPECT_EQ(0, occurrence_count) << "node.name()=" << node.name();
      }
      if (str_util::EndsWith(node.name(), "expect_remains")) {
        EXPECT_EQ(1, occurrence_count) << "node.name()=" << node.name();
      }
    }
  }

  void TestTransformRegistration() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    EXPECT_EQ(1, graph_def.node().size());
    TF_ASSERT_OK(TransformGraph({}, {}, {{"test_empty_graph_transform", {}}},
                                &graph_def));
    EXPECT_EQ(0, graph_def.node().size());

    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    Status no_such_status =
        TransformGraph({}, {}, {{"test_no_such_transform", {}}}, &graph_def);
    EXPECT_TRUE(absl::StrContains(no_such_status.ToString(), "not recognized"));
  }

  void TestParseTransformParameters() {
    TransformParameters params_list;

    TF_EXPECT_OK(ParseTransformParameters("foo", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());

    TF_EXPECT_OK(ParseTransformParameters("foo bar", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());

    TF_EXPECT_OK(ParseTransformParameters("foo() bar()", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());

    TF_EXPECT_OK(
        ParseTransformParameters("foo(bob_something=sue)", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("bob_something"));
    EXPECT_EQ(1, params_list[0].second["bob_something"].size());
    EXPECT_EQ("sue", params_list[0].second["bob_something"][0]);

    TF_EXPECT_OK(ParseTransformParameters("bar(a=1, b=2, a=3)", &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("bar", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("a"));
    EXPECT_EQ(2, params_list[0].second["a"].size());
    EXPECT_EQ("1", params_list[0].second["a"][0]);
    EXPECT_EQ("3", params_list[0].second["a"][1]);
    EXPECT_EQ(1, params_list[0].second.count("b"));
    EXPECT_EQ(1, params_list[0].second["b"].size());
    EXPECT_EQ("2", params_list[0].second["b"][0]);

    TF_EXPECT_OK(ParseTransformParameters("bar(a=\"1\", b=\"1,2,3\", a=3)",
                                          &params_list));
    EXPECT_EQ(1, params_list.size());
    EXPECT_EQ("bar", params_list[0].first);
    EXPECT_EQ(1, params_list[0].second.count("a"));
    EXPECT_EQ(2, params_list[0].second["a"].size());
    EXPECT_EQ("1", params_list[0].second["a"][0]);
    EXPECT_EQ("3", params_list[0].second["a"][1]);
    EXPECT_EQ(1, params_list[0].second.count("b"));
    EXPECT_EQ(1, params_list[0].second["b"].size());
    EXPECT_EQ("1,2,3", params_list[0].second["b"][0]);
  }

  void TestParseEscapedNewline() {
    // This sequence of characters caused an infinite loop in the parser, which
    // is responsible for the hang mentioned in
    // https://github.com/tensorflow/tensorflow/issues/7150
    TransformParameters params_list;
    ParseTransformParameters("\\\n", &params_list).IgnoreError();
    EXPECT_EQ(0, params_list.size());
  }

  void TestParseExtraSpaces() {
    TransformParameters params_list;
    ParseTransformParameters(" ", &params_list).IgnoreError();
    EXPECT_EQ(0, params_list.size());

    TF_EXPECT_OK(ParseTransformParameters("  foo bar \\\n", &params_list));
    EXPECT_EQ(2, params_list.size());
    EXPECT_EQ("foo", params_list[0].first);
    EXPECT_TRUE(params_list[0].second.empty());
    EXPECT_EQ("bar", params_list[1].first);
    EXPECT_TRUE(params_list[1].second.empty());
  }

  void TestShouldIgnoreErrors() {
    bool ignore_errors;
    TF_EXPECT_OK(
        ShouldIgnoreErrors({{"ignore_errors", {"true"}}}, &ignore_errors));
    EXPECT_TRUE(ignore_errors);

    TF_EXPECT_OK(
        ShouldIgnoreErrors({{"ignore_errors", {"false"}}}, &ignore_errors));
    EXPECT_FALSE(ignore_errors);

    TF_EXPECT_OK(ShouldIgnoreErrors({}, &ignore_errors));
    EXPECT_FALSE(ignore_errors);

    EXPECT_FALSE(
        ShouldIgnoreErrors({{"ignore_errors", {"foo"}}}, &ignore_errors).ok());
  }
};

TEST_F(TransformGraphTest, TestConstantFolding) { TestConstantFolding(); }

TEST_F(TransformGraphTest, TestTransformRegistration) {
  TestTransformRegistration();
}

TEST_F(TransformGraphTest, TestParseTransformParameters) {
  TestParseTransformParameters();
}

TEST_F(TransformGraphTest, TestParseEscapedNewline) {
  TestParseEscapedNewline();
}

TEST_F(TransformGraphTest, TestShouldIgnoreErrors) { TestShouldIgnoreErrors(); }

}  // namespace graph_transforms
}  // namespace tensorflow
