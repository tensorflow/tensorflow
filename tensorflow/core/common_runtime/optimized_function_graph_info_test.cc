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
#include "tensorflow/core/common_runtime/optimized_function_graph_info.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "third_party/protobuf/text_format.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/optimized_function_graph.pb.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::proto::Partially;
using ::tsl::testing::StatusIs;

REGISTER_OP("OneOutput").Output("y: float");
REGISTER_OP("OneInputTwoOutputs")
    .Input("x: float")
    .Output("y: float")
    .Output("z: float");

// Test function graph name.
constexpr absl::string_view kFunctionName = "test_func";
// Test library proto content.
constexpr absl::string_view kLibraryPb =
    R"pb(library {
           function {
             signature {
               name: "NonZero"
               input_arg { name: "x" type_attr: "T" }
               output_arg { name: "y" type_attr: "T" }
               attr {
                 name: "T"
                 type: "type"
                 allowed_values {
                   list {
                     type: DT_FLOAT
                     type: DT_DOUBLE
                     type: DT_INT32
                     type: DT_INT64
                     type: DT_STRING
                   }
                 }
               }
             }
             node_def {
               name: "y"
               op: "Identity"
               input: "x"
               attr {
                 key: "T"
                 value { placeholder: "T" }
               }
             }
             ret { key: "y" value: "y:output:0" }
           }
         })pb";

// Creates a simple graph with one trivial node.
absl::StatusOr<OptimizedFunctionGraphInfo>
CreateSimpleOptimizedFunctionGraphInfo() {
  NodeDef node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder("A", "OneOutput").Finalize(&node_def));
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status status;
  graph->AddNode(node_def, &status);
  TF_RETURN_IF_ERROR(status);

  // Create a simple library with one function.
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), FunctionDefLibrary());
  TF_RETURN_IF_ERROR(lib_def.AddFunctionDef(test::function::NonZero()));

  // Construct an OptimizedFunctionGraphInfo.
  return OptimizedFunctionGraphInfo(
      std::string(kFunctionName), std::move(graph), std::move(lib_def),
      {{"A", "B"}}, {DT_FLOAT, DT_DOUBLE}, 1, 5, OptimizedFunctionGraph::JIT);
}

TEST(OptimizedFunctionGraphUtilsTest, ToProtoProducesCorrectResult) {
  TF_ASSERT_OK_AND_ASSIGN(OptimizedFunctionGraphInfo test_info,
                          CreateSimpleOptimizedFunctionGraphInfo());

  const OptimizedFunctionGraph test_result =
      OptimizedFunctionGraphInfo::ToProto(test_info);
  EXPECT_THAT(test_result,
              Partially(EqualsProto(absl::Substitute(
                  R"pb(
                    name: "test_func"
                    function_graph { node { name: "A" op: "OneOutput" } $0 }
                    node_name_to_control_ret { key: "A" value: "B" }
                    ret_types: DT_FLOAT
                    ret_types: DT_DOUBLE
                    num_return_nodes: 1
                    source: JIT
                    optimization_time_usecs: 5
                  )pb",
                  kLibraryPb))));
}

TEST(OptimizedFunctionGraphUtilsTest,
     FromProtoProducesReturnsErrorIfGraphInvalid) {
  OptimizedFunctionGraph proto;
  // Invalid proto because no device specified for node B.
  proto2::TextFormat::ParseFromString(
      R"pb(
        name: "test_func",
        function_graph { node { name: 'B' op: 'OneOutput' } $0 }
      )pb",
      &proto);

  EXPECT_THAT(OptimizedFunctionGraphInfo::FromProto(std::move(proto)),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       "Node 'B' is missing a device specification"));
}

TEST(OptimizedFunctionGraphUtilsTest, FromProtoProducesCorrectResult) {
  OptimizedFunctionGraph proto;
  proto2::TextFormat::ParseFromString(
      absl::Substitute(
          R"pb(
            name: "test_func",
            function_graph {
              node { name: 'B' op: 'OneInputTwoOutputs' device: ':CPU' } $0
            }
            node_name_to_control_ret { key: "B" value: "A" }
            ret_types: DT_FLOAT
            ret_types: DT_DOUBLE
            ret_types: DT_BOOL
            num_return_nodes: 2
            optimization_time_usecs: 15
            source: 1
          )pb",
          kLibraryPb),
      &proto);

  const absl::StatusOr<OptimizedFunctionGraphInfo> test_result =
      OptimizedFunctionGraphInfo::FromProto(std::move(proto));
  TF_EXPECT_OK(test_result.status());
  // Compare graph.
  GraphDef test_result_graph_def;
  test_result->function_graph->ToGraphDef(&test_result_graph_def);
  EXPECT_THAT(test_result_graph_def,
              Partially(EqualsProto(R"pb(node {
                                           name: 'B'
                                           op: 'OneInputTwoOutputs'
                                           device: ':CPU'
                                         })pb")));
  // The lib_def in graph is already cleared.
  EXPECT_EQ(test_result->function_graph->flib_def().Find("NonZero"), nullptr);
  // The function should be found in result's lib_def.
  EXPECT_NE(test_result->lib_def.Find("NonZero"), nullptr);
  EXPECT_THAT(test_result->ret_types,
              ElementsAre(DT_FLOAT, DT_DOUBLE, DT_BOOL));
  EXPECT_THAT(test_result->node_name_to_control_ret,
              UnorderedElementsAre(Pair("B", "A")));
  EXPECT_EQ(test_result->num_return_nodes, 2);
  EXPECT_EQ(test_result->optimization_duration_usecs, 15);
  EXPECT_EQ(test_result->optimization_source, OptimizedFunctionGraph::AOT);
}

TEST(OptimizedFunctionGraphUtilsTest,
     FromProtoProducesCorrectResultWithFunctionCall) {
  OptimizedFunctionGraph proto;
  proto2::TextFormat::ParseFromString(
      absl::Substitute(
          R"pb(
            name: "test_func",
            function_graph {
              node {
                name: 'B'
                op: 'NonZero'
                device: ':CPU'
                attr {
                  key: "T"
                  value { type: DT_FLOAT }
                }
              } $0
            }
            node_name_to_control_ret { key: "B" value: "A" }
            ret_types: DT_FLOAT
            ret_types: DT_DOUBLE
            ret_types: DT_BOOL
            num_return_nodes: 2
            optimization_time_usecs: 15
            source: 1
          )pb",
          kLibraryPb),
      &proto);

  const absl::StatusOr<OptimizedFunctionGraphInfo> test_result =
      OptimizedFunctionGraphInfo::FromProto(std::move(proto));
  TF_EXPECT_OK(test_result.status());
  // Compare graph.
  GraphDef test_result_graph_def;
  test_result->function_graph->ToGraphDef(&test_result_graph_def);
  EXPECT_THAT(test_result_graph_def,
              Partially(EqualsProto(R"pb(node {
                                           name: 'B'
                                           op: 'NonZero'
                                           device: ':CPU'
                                           attr {
                                             key: "T"
                                             value { type: DT_FLOAT }
                                           }
                                         })pb")));
  // The lib_def in graph is already cleared.
  EXPECT_EQ(test_result->function_graph->flib_def().Find("NonZero"), nullptr);
  // The function should be found in result's lib_def.
  EXPECT_NE(test_result->lib_def.Find("NonZero"), nullptr);
  EXPECT_THAT(test_result->ret_types,
              ElementsAre(DT_FLOAT, DT_DOUBLE, DT_BOOL));
  EXPECT_THAT(test_result->node_name_to_control_ret,
              UnorderedElementsAre(Pair("B", "A")));
  EXPECT_EQ(test_result->num_return_nodes, 2);
  EXPECT_EQ(test_result->optimization_duration_usecs, 15);
  EXPECT_EQ(test_result->optimization_source, OptimizedFunctionGraph::AOT);
}

TEST(OptimizedFunctionGraphUtilsTest, MoveTest) {
  TF_ASSERT_OK_AND_ASSIGN(OptimizedFunctionGraphInfo test_info,
                          CreateSimpleOptimizedFunctionGraphInfo());

  OptimizedFunctionGraphInfo moved_result = std::move(test_info);

  // Compare graph.
  GraphDef moved_result_graph_def;
  moved_result.function_graph->ToGraphDef(&moved_result_graph_def);
  EXPECT_EQ(moved_result.name, kFunctionName);
  EXPECT_THAT(
      moved_result_graph_def,
      Partially(EqualsProto(R"pb(node { name: 'A' op: 'OneOutput' })pb")));
  // The function should be found in result's lib_def.
  EXPECT_NE(moved_result.lib_def.Find("NonZero"), nullptr);
  EXPECT_THAT(moved_result.ret_types, ElementsAre(DT_FLOAT, DT_DOUBLE));
  EXPECT_THAT(moved_result.node_name_to_control_ret,
              UnorderedElementsAre(Pair("A", "B")));
  EXPECT_EQ(moved_result.num_return_nodes, 1);
  EXPECT_EQ(moved_result.optimization_duration_usecs, 5);
  EXPECT_EQ(moved_result.optimization_source, OptimizedFunctionGraph::JIT);
}

}  // namespace
}  // namespace tensorflow
