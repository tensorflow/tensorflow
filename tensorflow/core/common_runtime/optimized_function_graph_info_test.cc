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
#include "absl/strings/substitute.h"
#include "third_party/protobuf/text_format.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/optimized_function_graph.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/test.h"

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

TEST(OptimizedFunctionGraphUtilsTest, ToProtoProducesCorrectResult) {
  // Create a simple graph with one trivial node.
  NodeDef node_def;
  TF_ASSERT_OK(NodeDefBuilder("A", "OneOutput").Finalize(&node_def));
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Status status;
  graph->AddNode(node_def, &status);
  TF_ASSERT_OK(status);

  // Create a simple library with one function.
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_ASSERT_OK(lib_def.AddFunctionDef(test::function::NonZero()));

  // Construct an OptimizedFunctionGraphInfo.
  OptimizedFunctionGraphInfo test_info{
      std::string(kFunctionName), std::move(graph),
      std::move(lib_def),         {{"A", "B"}},
      {DT_FLOAT, DT_DOUBLE},      1};

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

  EXPECT_THAT(OptimizedFunctionGraphInfo::FromProto(proto),
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
          )pb",
          kLibraryPb),
      &proto);

  const StatusOr<OptimizedFunctionGraphInfo> test_result =
      OptimizedFunctionGraphInfo::FromProto(proto);
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
}

}  // namespace
}  // namespace tensorflow
