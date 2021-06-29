/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

MATCHER_P2(IsStatus, error_code, error_message, "") {
  return arg.code() == error_code &&
         absl::StrContains(arg.error_message(), error_message);
}

Status RunGraph(const Graph& graph,
                const std::vector<std::string>& output_tensor_names,
                std::vector<Tensor>* output_tensors) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  SessionOptions session_options;
  std::unique_ptr<Session> session(NewSession(session_options));
  TF_RETURN_IF_ERROR(session->Create(graph_def));
  RunOptions run_options;
  return session->Run(run_options, /*inputs=*/{}, output_tensor_names,
                      /*target_tensor_names=*/{}, output_tensors,
                      /*run_metadata=*/nullptr);
}

TEST(ReadVariableXlaSplitNDOpTest, VariableMissing) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<int32>::value;
  const TensorShape input_shape({4, 4});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", data_type)
                   .Attr("shape", input_shape)
                   .Finalize(&graph, &var_handle));
  Node* xla_op = nullptr;
  const std::vector<int32> num_splits = {2, 2};
  const int num_outputs = 4;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "ReadVariableXlaSplitND")
                   .Input(var_handle)
                   .Attr("num_splits", num_splits)
                   .Attr("T", data_type)
                   .Attr("N", num_outputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, {xla_op->name()}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "cannot be found"));
}

TEST(ReadVariableXlaSplitNDOpTest, DTypeInvalid) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<int32>::value;
  const TensorShape input_shape({4, 4});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", data_type)
                   .Attr("shape", input_shape)
                   .Finalize(&graph, &var_handle));
  Tensor input_tensor(data_type, input_shape);
  test::FillIota<int32>(&input_tensor, /*val=*/0);
  Node* input = test::graph::Constant(&graph, input_tensor);
  Node* assign_var = nullptr;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("assign_var"), "AssignVariableOp")
                   .Input(var_handle)
                   .Input(input)
                   .Attr("dtype", data_type)
                   .Finalize(&graph, &assign_var));
  Node* xla_op = nullptr;
  const std::vector<int32> num_splits = {2, 2};
  const int num_outputs = 4;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "ReadVariableXlaSplitND")
                   .Input(var_handle)
                   .ControlInput(assign_var)
                   .Attr("num_splits", num_splits)
                   .Attr("T", DataTypeToEnum<float>::value)
                   .Attr("N", num_outputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, {xla_op->name()}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "'T' must match 'resource'"));
}

Status CreateTensorGraph(const TensorShape& input_shape,
                         absl::Span<const int32> num_splits,
                         absl::Span<const int32> paddings,
                         const int num_outputs, Graph* graph,
                         std::vector<std::string>* output_tensor_names) {
  DataType data_type = DataTypeToEnum<int32>::value;
  Tensor input_tensor(data_type, input_shape);
  test::FillIota<int32>(&input_tensor, /*val=*/0);
  Node* input = test::graph::Constant(graph, input_tensor);
  Node* xla_op = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("xla_op"), "XlaSplitND")
                         .Input(input)
                         .Attr("num_splits", num_splits)
                         .Attr("paddings", paddings)
                         .Attr("T", data_type)
                         .Attr("N", num_outputs)
                         .Finalize(graph, &xla_op));
  output_tensor_names->reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_tensor_names->push_back(absl::StrCat(xla_op->name(), ":", i));
  }
  return Status::OK();
}

Status CreateResourceGraph(const TensorShape& input_shape,
                           absl::Span<const int32> num_splits,
                           absl::Span<const int32> paddings,
                           const int num_outputs, Graph* graph,
                           std::vector<std::string>* output_tensor_names) {
  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<int32>::value;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("var_handle"), "VarHandleOp")
                         .Attr("dtype", data_type)
                         .Attr("shape", input_shape)
                         .Finalize(graph, &var_handle));
  Tensor input_tensor(data_type, input_shape);
  test::FillIota<int32>(&input_tensor, /*val=*/0);
  Node* input = test::graph::Constant(graph, input_tensor);
  Node* assign_var = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName("assign_var"), "AssignVariableOp")
          .Input(var_handle)
          .Input(input)
          .Attr("dtype", data_type)
          .Finalize(graph, &assign_var));
  Node* xla_op = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName("xla_op"), "ReadVariableXlaSplitND")
          .Input(var_handle)
          .ControlInput(assign_var)
          .Attr("num_splits", num_splits)
          .Attr("paddings", paddings)
          .Attr("T", data_type)
          .Attr("N", num_outputs)
          .Finalize(graph, &xla_op));
  output_tensor_names->reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_tensor_names->push_back(absl::StrCat(xla_op->name(), ":", i));
  }
  return Status::OK();
}

struct XlaSplitNDTestParam {
  std::string name;
  std::function<Status(const TensorShape&, absl::Span<const int32>,
                       absl::Span<const int32>, const int num_outputs, Graph*,
                       std::vector<std::string>*)>
      graph_creator;
};

using XlaSplitNDOpTest = ::testing::TestWithParam<XlaSplitNDTestParam>;

TEST_P(XlaSplitNDOpTest, SplitDimensionNegative) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1, 1, 1});
  const std::vector<int32> num_splits = {1, -1, 1};
  const std::vector<int32> paddings;
  const int num_outputs = 1;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT,
                       "index 1 must be positive, but got -1"));
}

TEST_P(XlaSplitNDOpTest, NumOutputsMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2});
  const std::vector<int32> num_splits = {2};
  const std::vector<int> paddings;
  const int num_outputs = 1;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "'N' must match number of slices 2"));
}

TEST_P(XlaSplitNDOpTest, PaddingsLengthMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings = {0};
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "length 2, but got 1"));
}

TEST_P(XlaSplitNDOpTest, PaddingsNegative) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings = {0, -1};
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "non-negative, but got -1 at index 1"));
}

TEST_P(XlaSplitNDOpTest, InputRank0) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({});
  const std::vector<int32> num_splits = {2};
  const std::vector<int32> paddings;
  const int num_outputs = 2;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "range (0, 8], but got 0"));
}

TEST_P(XlaSplitNDOpTest, InputRank9) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2, 2, 2, 2, 2, 2, 2, 2});
  const std::vector<int32> num_splits(9, 2);
  const std::vector<int32> paddings;
  const int num_outputs = 512;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "range (0, 8], but got 9"));
}

TEST_P(XlaSplitNDOpTest, InputRankMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2});
  const std::vector<int32> num_splits = {2, 2, 2};
  const std::vector<int32> paddings;
  const int num_outputs = 8;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "length 3, but got rank 2"));
}

TEST_P(XlaSplitNDOpTest, DimNotEvenlySplit) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({4, 2});
  const std::vector<int32> num_splits = {3, 2};
  const std::vector<int32> paddings;
  const int num_outputs = 6;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "divisible by 'num_splits' 3"));
}

TEST_P(XlaSplitNDOpTest, DimWithPaddingNotEvenlySplit) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({4, 2});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings = {0, 1};
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "divisible by 'num_splits' 2"));
}

TEST_P(XlaSplitNDOpTest, NoSplits) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2, 2});
  const std::vector<int32> num_splits = {1, 1, 1};
  const std::vector<int> paddings;
  const int num_outputs = 1;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 2, 3, 4, 5, 6, 7}, TensorShape({2, 2, 2})));
}

TEST_P(XlaSplitNDOpTest, NoSplitsWithPadding) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 1, 1});
  const std::vector<int32> num_splits = {1, 1, 1};
  const std::vector<int> paddings = {0, 1, 1};
  const int num_outputs = 1;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  std::vector<int32> expected_values(3 * 3 * 3);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 0, 0, 0, 1, 0, 0, 0}, TensorShape({2, 2, 2})));
}

TEST_P(XlaSplitNDOpTest, SliceDimPartialPadding) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({3, 3});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings = {1, 1};
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, &output_tensors));
  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 3, 4}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[1],
      test::AsTensor<int32>({2, 0, 5, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[2],
      test::AsTensor<int32>({6, 7, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[3],
      test::AsTensor<int32>({8, 0, 0, 0}, TensorShape({2, 2})));
}

TEST_P(XlaSplitNDOpTest, SliceDimCompletePadding) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 1});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings = {2, 3};
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, &output_tensors));
  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 0, 1, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[1],
      test::AsTensor<int32>({0, 0, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[2],
      test::AsTensor<int32>({0, 0, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[3],
      test::AsTensor<int32>({0, 0, 0, 0}, TensorShape({2, 2})));
}

INSTANTIATE_TEST_SUITE_P(
    XlaSplitNDOpTest, XlaSplitNDOpTest,
    ::testing::ValuesIn<XlaSplitNDTestParam>(
        {{"Tensor", CreateTensorGraph}, {"Resource", CreateResourceGraph}}),
    [](const ::testing::TestParamInfo<XlaSplitNDOpTest::ParamType>& info) {
      return info.param.name;
    });

struct RankedXlaSplitNDTestParam {
  std::string name;
  int rank = 0;
  std::function<Status(const TensorShape&, absl::Span<const int32>,
                       absl::Span<const int32>, const int num_outputs, Graph*,
                       std::vector<std::string>*)>
      graph_creator;
};

class RankedXlaSplitNDOpTest
    : public ::testing::TestWithParam<RankedXlaSplitNDTestParam> {};

TEST_P(RankedXlaSplitNDOpTest, TestSubscriptRank) {
  const int rank = GetParam().rank;
  const std::vector<int32> num_splits(rank, 2);

  Graph graph(OpRegistry::Global());
  const TensorShape input_shape(std::vector<int64>(rank, 2));
  const std::vector<int32> paddings;
  const int num_outputs = 2 << (rank - 1);
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, &output_tensors));
  ASSERT_EQ(output_tensors.size(), num_outputs);
  TensorShape output_shape(std::vector<int64>(rank, 1));
  for (int i = 0; i < num_outputs; ++i) {
    test::ExpectTensorEqual<int32>(output_tensors[i],
                                   test::AsTensor<int32>({i}, output_shape));
  }
}

INSTANTIATE_TEST_SUITE_P(
    RankedXlaSplitNDOpTest, RankedXlaSplitNDOpTest,
    ::testing::ValuesIn<RankedXlaSplitNDTestParam>(
        {{"TensorRanked1", 1, CreateTensorGraph},
         {"TensorRanked2", 2, CreateTensorGraph},
         {"TensorRanked3", 3, CreateTensorGraph},
         {"TensorRanked4", 4, CreateTensorGraph},
         {"TensorRanked5", 5, CreateTensorGraph},
         {"TensorRanked6", 6, CreateTensorGraph},
         {"TensorRanked7", 7, CreateTensorGraph},
         {"TensorRanked8", 8, CreateTensorGraph},
         {"ResourceRanked1", 1, CreateResourceGraph},
         {"ResourceRanked2", 2, CreateResourceGraph},
         {"ResourceRanked3", 3, CreateResourceGraph},
         {"ResourceRanked4", 4, CreateResourceGraph},
         {"ResourceRanked5", 5, CreateResourceGraph},
         {"ResourceRanked6", 6, CreateResourceGraph},
         {"ResourceRanked7", 7, CreateResourceGraph},
         {"ResourceRanked8", 8, CreateResourceGraph}}),
    [](const ::testing::TestParamInfo<RankedXlaSplitNDOpTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
}  // namespace tensorflow
