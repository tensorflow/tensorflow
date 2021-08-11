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
                const std::vector<std::string>& target_tensor_names,
                std::vector<Tensor>* output_tensors) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  SessionOptions session_options;
  std::unique_ptr<Session> session(NewSession(session_options));
  TF_RETURN_IF_ERROR(session->Create(graph_def));
  RunOptions run_options;
  return session->Run(run_options, /*inputs=*/{}, output_tensor_names,
                      target_tensor_names, output_tensors,
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
  EXPECT_THAT(RunGraph(graph, /*output_tensor_names=*/{xla_op->name()},
                       /*target_tensor_names=*/{}, &output_tensors),
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
  EXPECT_THAT(RunGraph(graph, /*output_tensor_names=*/{xla_op->name()},
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "'T' must match 'resource'"));
}

Status CreateSplitTensorGraph(const TensorShape& input_shape,
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

Status CreateSplitResourceGraph(const TensorShape& input_shape,
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

TEST_P(XlaSplitNDOpTest, SplitDimensionZero) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1, 1, 1});
  const std::vector<int32> num_splits = {1, 1, 0};
  const std::vector<int32> paddings;
  const int num_outputs = 1;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
               &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "index 2 must be positive, but got 0"));
}

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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
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
      RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
               &output_tensors),
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
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
      RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
               &output_tensors),
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "range (0, 8], but got 9"));
}

TEST_P(XlaSplitNDOpTest, InputRankSplitMismatch) {
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
              IsStatus(error::INVALID_ARGUMENT,
                       "'num_splits' length 3, but got rank 2"));
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
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
  EXPECT_THAT(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                       &output_tensors),
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
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
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
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  std::vector<int32> expected_values(3 * 3 * 3);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 0, 0, 0, 1, 0, 0, 0}, TensorShape({2, 2, 2})));
}

TEST_P(XlaSplitNDOpTest, SplitNoPadding) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({4, 4});
  const std::vector<int32> num_splits = {2, 2};
  const std::vector<int32> paddings;
  const int num_outputs = 4;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 4, 5}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[1],
      test::AsTensor<int32>({2, 3, 6, 7}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[2],
      test::AsTensor<int32>({8, 9, 12, 13}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32>(
      output_tensors[3],
      test::AsTensor<int32>({10, 11, 14, 15}, TensorShape({2, 2})));
}

TEST_P(XlaSplitNDOpTest, SplitPartialPadding) {
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
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
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

TEST_P(XlaSplitNDOpTest, SplitCompletePadding) {
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
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
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
        {{"Tensor", CreateSplitTensorGraph},
         {"Resource", CreateSplitResourceGraph}}),
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
  const TensorShape input_shape(std::vector<int64_t>(rank, 2));
  const std::vector<int32> paddings;
  const int num_outputs = 2 << (rank - 1);
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_splits, paddings,
                                        num_outputs, &graph,
                                        &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), num_outputs);
  TensorShape output_shape(std::vector<int64_t>(rank, 1));
  for (int i = 0; i < num_outputs; ++i) {
    test::ExpectTensorEqual<int32>(output_tensors[i],
                                   test::AsTensor<int32>({i}, output_shape));
  }
}

INSTANTIATE_TEST_SUITE_P(
    RankedXlaSplitNDOpTest, RankedXlaSplitNDOpTest,
    ::testing::ValuesIn<RankedXlaSplitNDTestParam>(
        {{"TensorRanked1", 1, CreateSplitTensorGraph},
         {"TensorRanked2", 2, CreateSplitTensorGraph},
         {"TensorRanked3", 3, CreateSplitTensorGraph},
         {"TensorRanked4", 4, CreateSplitTensorGraph},
         {"TensorRanked5", 5, CreateSplitTensorGraph},
         {"TensorRanked6", 6, CreateSplitTensorGraph},
         {"TensorRanked7", 7, CreateSplitTensorGraph},
         {"TensorRanked8", 8, CreateSplitTensorGraph},
         {"ResourceRanked1", 1, CreateSplitResourceGraph},
         {"ResourceRanked2", 2, CreateSplitResourceGraph},
         {"ResourceRanked3", 3, CreateSplitResourceGraph},
         {"ResourceRanked4", 4, CreateSplitResourceGraph},
         {"ResourceRanked5", 5, CreateSplitResourceGraph},
         {"ResourceRanked6", 6, CreateSplitResourceGraph},
         {"ResourceRanked7", 7, CreateSplitResourceGraph},
         {"ResourceRanked8", 8, CreateSplitResourceGraph}}),
    [](const ::testing::TestParamInfo<RankedXlaSplitNDOpTest::ParamType>&
           info) { return info.param.name; });

TEST(AssignVariableXlaConcatNDOpTest, HandleDTypeInvalid) {
  Graph graph(OpRegistry::Global());
  Node* var_handle = nullptr;
  DataType handle_dtype = DataTypeToEnum<int32>::value;
  PartialTensorShape handle_shape;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", handle_dtype)
                   .Attr("shape", handle_shape)
                   .Finalize(&graph, &var_handle));
  DataType update_data_type = DataTypeToEnum<float>::value;
  const TensorShape update_input_shape({4, 4});
  Tensor update_input_tensor(update_data_type, update_input_shape);
  test::FillIota<float>(&update_input_tensor, /*val=*/0.f);
  Node* update_input = test::graph::Constant(&graph, update_input_tensor);
  Node* xla_op = nullptr;
  const std::vector<int32> num_concats = {1, 1};
  const int num_inputs = 1;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "AssignVariableXlaConcatND")
                   .Input(var_handle)
                   .Input(std::vector<NodeBuilder::NodeOut>{update_input})
                   .Attr("num_concats", num_concats)
                   .Attr("T", update_data_type)
                   .Attr("N", num_inputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, /*output_tensor_names=*/{},
               /*target_tensor_names=*/{xla_op->name()}, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "dtype int32, but got float"));
}

TEST(AssignVariableXlaConcatNDOpTest, TensorDTypeInvalid) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType handle_dtype = DataTypeToEnum<float>::value;
  PartialTensorShape handle_shape;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", handle_dtype)
                   .Attr("shape", handle_shape)
                   .Finalize(&graph, &var_handle));

  DataType init_data_type = DataTypeToEnum<int32>::value;
  const TensorShape init_input_shape({4, 4});
  Tensor init_input_tensor(init_data_type, init_input_shape);
  test::FillIota<int32>(&init_input_tensor, /*val=*/0);
  Node* input = test::graph::Constant(&graph, init_input_tensor);

  Node* assign_var = nullptr;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("assign_var"), "AssignVariableOp")
                   .Input(var_handle)
                   .Input(input)
                   .Attr("dtype", init_data_type)
                   .Finalize(&graph, &assign_var));

  DataType update_data_type = DataTypeToEnum<float>::value;
  const TensorShape update_input_shape({4, 4});
  Tensor update_input_tensor(update_data_type, update_input_shape);
  test::FillIota<float>(&update_input_tensor, /*val=*/0.f);
  Node* update_input = test::graph::Constant(&graph, update_input_tensor);

  Node* xla_op = nullptr;
  const std::vector<int32> num_concats = {1, 1};
  const int num_inputs = 1;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "AssignVariableXlaConcatND")
                   .Input(var_handle)
                   .Input(std::vector<NodeBuilder::NodeOut>{update_input})
                   .ControlInput(assign_var)
                   .Attr("num_concats", num_concats)
                   .Attr("T", update_data_type)
                   .Attr("N", num_inputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, /*output_tensor_names=*/{},
               /*target_tensor_names=*/{xla_op->name()}, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "dtype int32, but got float"));
}

TEST(AssignVariableXlaConcatNDOpTest, HandleShapeIncompatible) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType handle_dtype = DataTypeToEnum<float>::value;
  PartialTensorShape handle_shape({});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", handle_dtype)
                   .Attr("shape", handle_shape)
                   .Finalize(&graph, &var_handle));

  DataType update_data_type = DataTypeToEnum<float>::value;
  const TensorShape update_input_shape({4, 4});
  Tensor update_input_tensor(update_data_type, update_input_shape);
  test::FillIota<float>(&update_input_tensor, /*val=*/0.f);
  Node* update_input = test::graph::Constant(&graph, update_input_tensor);

  Node* xla_op = nullptr;
  const std::vector<int32> num_concats = {1, 1};
  const int num_inputs = 1;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "AssignVariableXlaConcatND")
                   .Input(var_handle)
                   .Input(std::vector<NodeBuilder::NodeOut>{update_input})
                   .Attr("num_concats", num_concats)
                   .Attr("T", update_data_type)
                   .Attr("N", num_inputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, /*output_tensor_names=*/{},
               /*target_tensor_names=*/{xla_op->name()}, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "expected shape [4,4], but got []"));
}

TEST(AssignVariableXlaConcatNDOpTest, HandleShapeWithPaddingIncompatible) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType handle_dtype = DataTypeToEnum<float>::value;
  PartialTensorShape handle_shape({4, 4});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", handle_dtype)
                   .Attr("shape", handle_shape)
                   .Finalize(&graph, &var_handle));

  DataType update_data_type = DataTypeToEnum<float>::value;
  const TensorShape update_input_shape({4, 4});
  Tensor update_input_tensor(update_data_type, update_input_shape);
  test::FillIota<float>(&update_input_tensor, /*val=*/0.f);
  Node* update_input = test::graph::Constant(&graph, update_input_tensor);

  Node* xla_op = nullptr;
  const std::vector<int32> num_concats = {1, 1};
  const std::vector<int32> paddings = {1, 1};
  const int num_inputs = 1;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "AssignVariableXlaConcatND")
                   .Input(var_handle)
                   .Input(std::vector<NodeBuilder::NodeOut>{update_input})
                   .Attr("num_concats", num_concats)
                   .Attr("paddings", paddings)
                   .Attr("T", update_data_type)
                   .Attr("N", num_inputs)
                   .Finalize(&graph, &xla_op));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, /*output_tensor_names=*/{},
               /*target_tensor_names=*/{xla_op->name()}, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "expected shape [3,3], but got [4,4]"));
}

TEST(AssignVariableXlaConcatNDOpTest, AssignDifferentShape) {
  Graph graph(OpRegistry::Global());

  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<float>::value;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("var_handle"), "VarHandleOp")
                   .Attr("dtype", data_type)
                   .Attr("shape", PartialTensorShape({4, -1}))
                   .Finalize(&graph, &var_handle));

  const TensorShape init_input_shape({4, 2});
  Tensor init_input_tensor(data_type, init_input_shape);
  test::FillFn<float>(&init_input_tensor, [](int unused) { return -1.f; });
  Node* init_input = test::graph::Constant(&graph, init_input_tensor);

  Node* assign_var = nullptr;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("assign_var"), "AssignVariableOp")
                   .Input(var_handle)
                   .Input(init_input)
                   .Attr("dtype", data_type)
                   .Finalize(&graph, &assign_var));

  const TensorShape update_input_shape({4, 4});
  Tensor update_input_tensor(data_type, update_input_shape);
  test::FillIota<float>(&update_input_tensor, /*val=*/0.f);
  Node* update_input = test::graph::Constant(&graph, update_input_tensor);

  Node* xla_op = nullptr;
  const std::vector<int32> num_concats = {1, 1};
  const int num_inputs = 1;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("xla_op"), "AssignVariableXlaConcatND")
                   .Input(var_handle)
                   .Input(std::vector<NodeBuilder::NodeOut>{update_input})
                   .ControlInput(assign_var)
                   .Attr("num_concats", num_concats)
                   .Attr("T", data_type)
                   .Attr("N", num_inputs)
                   .Finalize(&graph, &xla_op));

  Node* read_var = nullptr;
  TF_ASSERT_OK(NodeBuilder(graph.NewName("read_var"), "ReadVariableOp")
                   .Input(var_handle)
                   .ControlInput(xla_op)
                   .Attr("dtype", data_type)
                   .Finalize(&graph, &read_var));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(
      graph, /*output_tensor_names=*/{absl::StrCat(read_var->name(), ":", 0)},
      /*target_tensor_names=*/{}, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorNear<float>(output_tensors[0], update_input_tensor,
                                /*atol=*/1e-6);
}

Status CreateConcatTensorGraph(absl::Span<const TensorShape> input_shapes,
                               absl::Span<const int32> num_concats,
                               absl::Span<const int32> paddings, Graph* graph,
                               std::vector<std::string>* output_tensor_names) {
  int32_t val = 0;
  DataType data_type = DataTypeToEnum<int32>::value;
  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(input_shapes.size());
  for (const TensorShape& input_shape : input_shapes) {
    Tensor input_tensor(data_type, input_shape);
    test::FillIota<int32>(&input_tensor, val);
    val += input_tensor.NumElements();
    inputs.push_back(test::graph::Constant(graph, input_tensor));
  }

  Node* xla_op = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("xla_op"), "XlaConcatND")
                         .Input(inputs)
                         .Attr("num_concats", num_concats)
                         .Attr("paddings", paddings)
                         .Attr("T", data_type)
                         .Attr("N", static_cast<int64_t>(input_shapes.size()))
                         .Finalize(graph, &xla_op));

  output_tensor_names->push_back(absl::StrCat(xla_op->name(), ":", 0));

  return Status::OK();
}

template <bool Init>
Status CreateConcatResourceGraph(
    absl::Span<const TensorShape> input_shapes,
    absl::Span<const int32> num_concats, absl::Span<const int32> paddings,
    Graph* graph, std::vector<std::string>* output_tensor_names) {
  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<int32>::value;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("var_handle"), "VarHandleOp")
                         .Attr("dtype", data_type)
                         .Attr("shape", PartialTensorShape())
                         .Finalize(graph, &var_handle));

  Node* assign_var = nullptr;
  if (Init) {
    Tensor init_input_tensor(data_type, input_shapes.front());
    test::FillFn<int32>(&init_input_tensor, [](int unused) { return -1; });
    Node* init_input = test::graph::Constant(graph, init_input_tensor);

    TF_RETURN_IF_ERROR(
        NodeBuilder(graph->NewName("assign_var"), "AssignVariableOp")
            .Input(var_handle)
            .Input(init_input)
            .Attr("dtype", data_type)
            .Finalize(graph, &assign_var));
  }

  int32_t val = 0;
  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(input_shapes.size());
  for (const TensorShape& input_shape : input_shapes) {
    Tensor input_tensor(data_type, input_shape);
    test::FillIota<int32>(&input_tensor, val);
    val += input_tensor.NumElements();
    inputs.push_back(test::graph::Constant(graph, input_tensor));
  }

  Node* xla_op = nullptr;
  NodeBuilder builder(graph->NewName("xla_op"), "AssignVariableXlaConcatND");
  builder.Input(var_handle);
  builder.Input(inputs);
  if (assign_var != nullptr) {
    builder.ControlInput(assign_var);
  }
  TF_RETURN_IF_ERROR(builder.Attr("num_concats", num_concats)
                         .Attr("paddings", paddings)
                         .Attr("T", data_type)
                         .Attr("N", static_cast<int64_t>(input_shapes.size()))
                         .Finalize(graph, &xla_op));

  Node* read_var = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("read_var"), "ReadVariableOp")
                         .Input(var_handle)
                         .ControlInput(xla_op)
                         .Attr("dtype", data_type)
                         .Finalize(graph, &read_var));

  output_tensor_names->push_back(absl::StrCat(read_var->name(), ":", 0));

  return Status::OK();
}

struct XlaConcatNDTestParam {
  std::string name;
  std::function<Status(absl::Span<const TensorShape>, absl::Span<const int32>,
                       absl::Span<const int32>, Graph*,
                       std::vector<std::string>*)>
      graph_creator;
};

using XlaConcatNDOpTest = ::testing::TestWithParam<XlaConcatNDTestParam>;

TEST_P(XlaConcatNDOpTest, ConcatDimensionZero) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1, 1, 1});
  const std::vector<int32> num_concats = {1, 1, 0};
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names,
               /*target_tensor_names=*/{}, &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "index 2 must be positive, but got 0"));
}

TEST_P(XlaConcatNDOpTest, ConcatDimensionNegative) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1, 1, 1});
  const std::vector<int32> num_splits = {1, -1, 1};
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_splits, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT,
                       "index 1 must be positive, but got -1"));
}

TEST_P(XlaConcatNDOpTest, NumInputsMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2});
  const std::vector<int32> num_concats = {2};
  const std::vector<int> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
               &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "'N' must match number of slices 2"));
}

TEST_P(XlaConcatNDOpTest, PaddingsLengthMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2});
  const std::vector<int32> num_concats = {1, 1};
  const std::vector<int32> paddings = {0};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "length 2, but got 1"));
}

TEST_P(XlaConcatNDOpTest, PaddingsNegative) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2});
  const std::vector<int32> num_concats = {1, 1};
  const std::vector<int32> paddings = {0, -1};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
               &output_tensors),
      IsStatus(error::INVALID_ARGUMENT, "non-negative, but got -1 at index 1"));
}

TEST_P(XlaConcatNDOpTest, InputRank0) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({});
  const std::vector<int32> num_concats;
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "range (0, 8], but got 0"));
}

TEST_P(XlaConcatNDOpTest, InputRank9) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1, 1, 1, 1, 1, 1, 1, 1, 1});
  const std::vector<int32> num_concats(9, 1);
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT, "range (0, 8], but got 9"));
}

TEST_P(XlaConcatNDOpTest, InputRankConcatMismatch) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1});
  const std::vector<int32> num_concats = {1, 1};
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT,
                       "'num_concats' length 2, but got rank 1"));
}

TEST_P(XlaConcatNDOpTest, DifferentShapedInputs) {
  Graph graph(OpRegistry::Global());
  const std::vector<TensorShape> input_shapes{{1}, {2}};
  const std::vector<int32> num_concats = {2};
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shapes, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(RunGraph(graph, output_tensor_names,
                       /*target_tensor_names=*/{}, &output_tensors),
              IsStatus(error::INVALID_ARGUMENT,
                       "same expected shape [1], but got [2] at index 1"));
}

TEST_P(XlaConcatNDOpTest, PaddingExceedsOutputDimSize) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({1});
  const std::vector<int32> num_concats = {1};
  const std::vector<int32> paddings = {2};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  EXPECT_THAT(
      RunGraph(graph, output_tensor_names,
               /*target_tensor_names=*/{}, &output_tensors),
      IsStatus(
          error::INVALID_ARGUMENT,
          "exceed expected output shape dimension 1 at index 0, but got 2"));
}

TEST_P(XlaConcatNDOpTest, NoConcats) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2, 2});
  const std::vector<int32> num_concats = {1, 1, 1};
  const std::vector<int> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 2, 3, 4, 5, 6, 7}, TensorShape({2, 2, 2})));
}

TEST_P(XlaConcatNDOpTest, NoConcatsWithPadding) {
  Graph graph(OpRegistry::Global());
  const TensorShape input_shape({2, 2, 2});
  const std::vector<int32> num_concats = {1, 1, 1};
  const std::vector<int> paddings = {1, 1, 1};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator({input_shape}, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names,
                        /*target_tensor_names=*/{}, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0], test::AsTensor<int32>({0}, TensorShape({1, 1, 1})));
}

TEST_P(XlaConcatNDOpTest, ConcatNoPadding) {
  Graph graph(OpRegistry::Global());
  const std::vector<TensorShape> input_shapes{{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  const std::vector<int32> num_concats = {2, 2};
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shapes, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names,
                        /*target_tensor_names=*/{}, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0], test::AsTensor<int32>({0, 1, 4, 5, 2, 3, 6, 7, 8, 9,
                                                12, 13, 10, 11, 14, 15},
                                               TensorShape({4, 4})));
}

TEST_P(XlaConcatNDOpTest, ConcatPartialPadding) {
  Graph graph(OpRegistry::Global());
  const std::vector<TensorShape> input_shapes{{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  const std::vector<int32> num_concats = {2, 2};
  const std::vector<int32> paddings = {1, 1};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shapes, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names,
                        /*target_tensor_names=*/{}, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 4, 2, 3, 6, 8, 9, 12}, TensorShape({3, 3})));
}

TEST_P(XlaConcatNDOpTest, ConcatCompletePadding) {
  Graph graph(OpRegistry::Global());
  const std::vector<TensorShape> input_shapes{{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  const std::vector<int32> num_concats = {2, 2};
  const std::vector<int32> paddings = {2, 2};
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shapes, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names,
                        /*target_tensor_names=*/{}, &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>({0, 1, 2, 3}, TensorShape({2, 2})));
}

INSTANTIATE_TEST_SUITE_P(
    XlaConcatNDOpTest, XlaConcatNDOpTest,
    ::testing::ValuesIn<XlaConcatNDTestParam>(
        {{"Tensor", CreateConcatTensorGraph},
         {"InitializedResource", CreateConcatResourceGraph<true>},
         {"UninitializedResource", CreateConcatResourceGraph<false>}}),
    [](const ::testing::TestParamInfo<XlaConcatNDOpTest::ParamType>& info) {
      return info.param.name;
    });

struct RankedXlaConcatNDTestParam {
  std::string name;
  int rank = 0;
  std::function<Status(absl::Span<const TensorShape>, absl::Span<const int32>,
                       absl::Span<const int32>, Graph*,
                       std::vector<std::string>*)>
      graph_creator;
};

class RankedXlaConcatNDOpTest
    : public ::testing::TestWithParam<RankedXlaConcatNDTestParam> {};

TEST_P(RankedXlaConcatNDOpTest, TestSubscriptRank) {
  const int rank = GetParam().rank;
  const std::vector<int32> num_concats(rank, 2);

  Graph graph(OpRegistry::Global());
  const int num_inputs = 2 << (rank - 1);
  const TensorShape base_input_shape(std::vector<int64_t>(rank, 1));
  const std::vector<TensorShape> input_shapes(num_inputs, base_input_shape);
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shapes, num_concats, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);
  std::vector<int32> expected_values(num_inputs);
  std::iota(expected_values.begin(), expected_values.end(), 0);
  test::ExpectTensorEqual<int32>(
      output_tensors[0],
      test::AsTensor<int32>(expected_values,
                            TensorShape(std::vector<int64_t>(rank, 2))));
}

INSTANTIATE_TEST_SUITE_P(
    RankedXlaConcatNDOpTest, RankedXlaConcatNDOpTest,
    ::testing::ValuesIn<RankedXlaConcatNDTestParam>(
        {{"TensorRanked1", 1, CreateConcatTensorGraph},
         {"TensorRanked2", 2, CreateConcatTensorGraph},
         {"TensorRanked3", 3, CreateConcatTensorGraph},
         {"TensorRanked4", 4, CreateConcatTensorGraph},
         {"TensorRanked5", 5, CreateConcatTensorGraph},
         {"TensorRanked6", 6, CreateConcatTensorGraph},
         {"TensorRanked7", 7, CreateConcatTensorGraph},
         {"TensorRanked8", 8, CreateConcatTensorGraph},
         {"InitializedResourceRanked1", 1, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked2", 2, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked3", 3, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked4", 4, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked5", 5, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked6", 6, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked7", 7, CreateConcatResourceGraph<true>},
         {"InitializedResourceRanked8", 8, CreateConcatResourceGraph<true>},
         {"UninitializedResourceRanked1", 1, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked2", 2, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked3", 3, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked4", 4, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked5", 5, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked6", 6, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked7", 7, CreateConcatResourceGraph<false>},
         {"UninitializedResourceRanked8", 8,
          CreateConcatResourceGraph<false>}}),
    [](const ::testing::TestParamInfo<RankedXlaConcatNDOpTest::ParamType>&
           info) { return info.param.name; });

Status CreateRoundtripTensorGraph(
    const TensorShape& input_shape, absl::Span<const int32> num_partitions,
    absl::Span<const int32> paddings, Graph* graph,
    std::vector<std::string>* output_tensor_names) {
  const int32_t num_partitions_size =
      std::accumulate(num_partitions.begin(), num_partitions.end(), 1,
                      std::multiplies<int32>());

  DataType data_type = DataTypeToEnum<int32>::value;
  Tensor input_tensor(data_type, input_shape);
  test::FillIota<int32>(&input_tensor, /*val=*/0);
  Node* input = test::graph::Constant(graph, input_tensor);

  Node* xla_split_op = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("xla_split_op"), "XlaSplitND")
                         .Input(input)
                         .Attr("num_splits", num_partitions)
                         .Attr("paddings", paddings)
                         .Attr("T", data_type)
                         .Attr("N", num_partitions_size)
                         .Finalize(graph, &xla_split_op));

  std::vector<NodeBuilder::NodeOut> concat_inputs;
  concat_inputs.reserve(num_partitions_size);
  for (int i = 0; i < num_partitions_size; ++i) {
    concat_inputs.push_back({xla_split_op, i});
  }

  Node* xla_concat_op = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("xla_concat_op"), "XlaConcatND")
                         .Input(concat_inputs)
                         .Attr("num_concats", num_partitions)
                         .Attr("paddings", paddings)
                         .Attr("T", data_type)
                         .Attr("N", num_partitions_size)
                         .Finalize(graph, &xla_concat_op));

  Node* equal = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("equal"), "Equal")
                         .Input(input)
                         .Input(xla_concat_op)
                         .Attr("T", data_type)
                         .Finalize(graph, &equal));

  output_tensor_names->push_back(absl::StrCat(equal->name(), ":", 0));

  return Status::OK();
}

Status CreateRoundtripResourceGraph(
    const TensorShape& input_shape, absl::Span<const int32> num_partitions,
    absl::Span<const int32> paddings, Graph* graph,
    std::vector<std::string>* output_tensor_names) {
  const int32_t num_partitions_size =
      std::accumulate(num_partitions.begin(), num_partitions.end(), 1,
                      std::multiplies<int32>());

  Node* var_handle = nullptr;
  DataType data_type = DataTypeToEnum<int32>::value;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("var_handle"), "VarHandleOp")
                         .Attr("dtype", data_type)
                         .Attr("shape", PartialTensorShape())
                         .Finalize(graph, &var_handle));

  Tensor input_tensor(data_type, input_shape);
  test::FillIota<int32>(&input_tensor, 0);
  Node* input = test::graph::Constant(graph, input_tensor);

  Node* assign_var = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName("assign_var"), "AssignVariableOp")
          .Input(var_handle)
          .Input(input)
          .Attr("dtype", data_type)
          .Finalize(graph, &assign_var));

  Node* xla_split_op = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName("xla_split_op"), "ReadVariableXlaSplitND")
          .Input(var_handle)
          .ControlInput(assign_var)
          .Attr("num_splits", num_partitions)
          .Attr("paddings", paddings)
          .Attr("T", data_type)
          .Attr("N", num_partitions_size)
          .Finalize(graph, &xla_split_op));

  std::vector<NodeBuilder::NodeOut> concat_inputs;
  concat_inputs.reserve(num_partitions_size);
  for (int i = 0; i < num_partitions_size; ++i) {
    concat_inputs.push_back({xla_split_op, i});
  }

  Node* xla_concat_op = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName("xla_op"), "AssignVariableXlaConcatND")
          .Input(var_handle)
          .Input(concat_inputs)
          .Attr("num_concats", num_partitions)
          .Attr("paddings", paddings)
          .Attr("T", data_type)
          .Attr("N", num_partitions_size)
          .Finalize(graph, &xla_concat_op));

  Node* read_var = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("read_var"), "ReadVariableOp")
                         .Input(var_handle)
                         .ControlInput(xla_concat_op)
                         .Attr("dtype", data_type)
                         .Finalize(graph, &read_var));

  Node* equal = nullptr;
  TF_RETURN_IF_ERROR(NodeBuilder(graph->NewName("equal"), "Equal")
                         .Input(input)
                         .Input(read_var)
                         .Attr("T", data_type)
                         .Finalize(graph, &equal));

  output_tensor_names->push_back(absl::StrCat(equal->name(), ":", 0));

  return Status::OK();
}

struct RoundtripXlaSplitConcatNDTestParam {
  std::string name;
  int rank = 0;
  std::function<Status(const TensorShape&, absl::Span<const int32>,
                       absl::Span<const int32>, Graph*,
                       std::vector<std::string>*)>
      graph_creator;
};

class RoundtripXlaSplitConcatNDTest
    : public ::testing::TestWithParam<RoundtripXlaSplitConcatNDTestParam> {};

template <typename T>
Tensor Constant(T v, TensorShape shape) {
  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

TEST_P(RoundtripXlaSplitConcatNDTest, NoPadding) {
  const int rank = GetParam().rank;
  const std::vector<int32> num_partitions(rank, 2);

  Graph graph(OpRegistry::Global());
  const TensorShape input_shape(std::vector<int64_t>(rank, 4));
  const std::vector<int32> paddings;
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_partitions, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);

  test::ExpectTensorEqual<bool>(
      output_tensors[0],
      Constant<bool>(true, TensorShape(std::vector<int64_t>(rank, 4))));
}

TEST_P(RoundtripXlaSplitConcatNDTest, PartialPadding) {
  const int rank = GetParam().rank;
  const std::vector<int32> num_partitions(rank, 2);

  Graph graph(OpRegistry::Global());
  const TensorShape input_shape(std::vector<int64_t>(rank, 4));
  const std::vector<int32> paddings(rank, 2);
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_partitions, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);

  test::ExpectTensorEqual<bool>(
      output_tensors[0],
      Constant<bool>(true, TensorShape(std::vector<int64_t>(rank, 4))));
}

TEST_P(RoundtripXlaSplitConcatNDTest, CompletePadding) {
  const int rank = GetParam().rank;
  const std::vector<int32> num_partitions(rank, 2);

  Graph graph(OpRegistry::Global());
  const TensorShape input_shape(std::vector<int64_t>(rank, 4));
  const std::vector<int32> paddings(rank, 4);
  std::vector<std::string> output_tensor_names;
  TF_ASSERT_OK(GetParam().graph_creator(input_shape, num_partitions, paddings,
                                        &graph, &output_tensor_names));

  std::vector<Tensor> output_tensors;
  TF_ASSERT_OK(RunGraph(graph, output_tensor_names, /*target_tensor_names=*/{},
                        &output_tensors));
  ASSERT_EQ(output_tensors.size(), 1);

  test::ExpectTensorEqual<bool>(
      output_tensors[0],
      Constant<bool>(true, TensorShape(std::vector<int64_t>(rank, 4))));
}

INSTANTIATE_TEST_SUITE_P(
    RoundtripXlaSplitConcatNDTest, RoundtripXlaSplitConcatNDTest,
    ::testing::ValuesIn<RoundtripXlaSplitConcatNDTestParam>(
        {{"TensorRanked1", 1, CreateRoundtripTensorGraph},
         {"TensorRanked2", 2, CreateRoundtripTensorGraph},
         {"TensorRanked3", 3, CreateRoundtripTensorGraph},
         {"TensorRanked4", 4, CreateRoundtripTensorGraph},
         {"TensorRanked5", 5, CreateRoundtripTensorGraph},
         {"TensorRanked6", 6, CreateRoundtripTensorGraph},
         {"TensorRanked7", 7, CreateRoundtripTensorGraph},
         {"TensorRanked8", 8, CreateRoundtripTensorGraph},
         {"ResourceRanked1", 1, CreateRoundtripResourceGraph},
         {"ResourceRanked2", 2, CreateRoundtripResourceGraph},
         {"ResourceRanked3", 3, CreateRoundtripResourceGraph},
         {"ResourceRanked4", 4, CreateRoundtripResourceGraph},
         {"ResourceRanked5", 5, CreateRoundtripResourceGraph},
         {"ResourceRanked6", 6, CreateRoundtripResourceGraph},
         {"ResourceRanked7", 7, CreateRoundtripResourceGraph},
         {"ResourceRanked8", 8, CreateRoundtripResourceGraph}}),
    [](const ::testing::TestParamInfo<RoundtripXlaSplitConcatNDTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
}  // namespace tensorflow
