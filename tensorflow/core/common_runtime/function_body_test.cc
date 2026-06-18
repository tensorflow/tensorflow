/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function_body.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace {

using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::StatusIs;

NodeDef GetNodeDef(
    absl::string_view name, absl::string_view op,
    absl::Span<const std::string> inputs,
    absl::Span<
        const std::pair<std::string, FunctionDefHelper::AttrValueWrapper>>
        attrs,
    bool set_full_type_def = false) {
  NodeDef node_def = test::function::NDef(name, op, inputs, attrs);
  if (!set_full_type_def) return node_def;

  FullTypeDef& experiment_type = *node_def.mutable_experimental_type();
  experiment_type.set_type_id(TFT_PRODUCT);
  experiment_type.add_args()->set_type_id(TFT_SHAPE_TENSOR);
  return node_def;
}

TEST(FunctionBodyTest, EmptyFunctionBody) {
  core::RefCountPtr<FunctionRecord> record(new FunctionRecord(
      FunctionDef(), /*stack_traces=*/{}, /*finalized=*/false));
  Graph* graph = new Graph(OpRegistry::Global());
  FunctionBody fbody(std::move(record), {}, {}, graph);

  EXPECT_THAT(fbody.record, Not(IsNull()));
  EXPECT_THAT(fbody.graph, Not(IsNull()));
  EXPECT_THAT(fbody.arg_types, IsEmpty());
  EXPECT_THAT(fbody.ret_types, IsEmpty());
  EXPECT_THAT(fbody.arg_nodes, IsEmpty());
  EXPECT_THAT(fbody.ret_nodes, IsEmpty());
  EXPECT_THAT(fbody.control_ret_nodes, IsEmpty());
  EXPECT_THAT(fbody.args_alloc_attrs, IsEmpty());
  EXPECT_THAT(fbody.rets_alloc_attrs, IsEmpty());
}

TEST(FunctionBodyTest, SimpleFunctionBody) {
  core::RefCountPtr<FunctionRecord> record(
      new FunctionRecord(test::function::XTimesTwoWithControlOutput(),
                         /*stack_traces=*/{}, /*finalized=*/false));
  Graph* graph = new Graph(OpRegistry::Global());
  TF_ASSERT_OK(graph->AddNode(
      GetNodeDef("x", FunctionLibraryDefinition::kArgOp, /*inputs=*/{},
                 /*attrs=*/{{"T", DT_INT32}, {"index", 0}})));
  TF_ASSERT_OK(graph->AddNode(
      GetNodeDef("y", FunctionLibraryDefinition::kRetOp, /*inputs=*/{},
                 /*attrs=*/{{"T", DT_INT32}, {"index", 0}})));
  TF_ASSERT_OK(graph->AddNode(GetNodeDef("dummy", "Const", /*inputs=*/{},
                                         /*attrs=*/{{"dtype", DT_INT32}})));
  FunctionBody fbody(std::move(record), {DT_INT32}, {DT_INT32}, graph);

  EXPECT_THAT(fbody.record, Not(IsNull()));
  EXPECT_THAT(fbody.graph, Not(IsNull()));
  EXPECT_THAT(fbody.arg_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.ret_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.arg_nodes,
              UnorderedElementsAre(Pointee(Property(
                  &Node::type_string, FunctionLibraryDefinition::kArgOp))));
  EXPECT_THAT(fbody.ret_nodes,
              UnorderedElementsAre(Pointee(Property(
                  &Node::type_string, FunctionLibraryDefinition::kRetOp))));
  EXPECT_THAT(fbody.control_ret_nodes,
              UnorderedElementsAre(Pointee(Property(&Node::name, "dummy"))));
  EXPECT_THAT(fbody.args_alloc_attrs, IsEmpty());
  EXPECT_THAT(fbody.rets_alloc_attrs, IsEmpty());
}

TEST(FunctionBodyTest, FunctionBodyFinalized) {
  core::RefCountPtr<FunctionRecord> record(
      new FunctionRecord(test::function::XTimesTwoWithControlOutput(),
                         /*stack_traces=*/{}, /*finalized=*/false));
  Graph* graph = new Graph(OpRegistry::Global());
  TF_ASSERT_OK(graph->AddNode(GetNodeDef(
      "x", FunctionLibraryDefinition::kArgOp, /*inputs=*/{},
      /*attrs=*/{{"T", DT_INT32}, {"index", 0}}, /*set_full_type_def=*/true)));
  TF_ASSERT_OK_AND_ASSIGN(
      Node * output_const,
      graph->AddNode(GetNodeDef("output_const", "Const", /*inputs=*/{},
                                /*attrs=*/{{"dtype", DT_INT32}},
                                /*set_full_type_def=*/true)));
  TF_ASSERT_OK_AND_ASSIGN(
      Node * ret_node,
      graph->AddNode(GetNodeDef("y", FunctionLibraryDefinition::kRetOp,
                                /*inputs=*/{"output_const"},
                                /*attrs=*/{{"T", DT_INT32}, {"index", 0}})));
  graph->AddEdge(output_const, 0, ret_node, 0);
  TF_ASSERT_OK(graph->AddNode(GetNodeDef("dummy", "Const", /*inputs=*/{},
                                         /*attrs=*/{{"dtype", DT_INT32}})));
  FunctionBody fbody(std::move(record), {DT_INT32}, {DT_INT32}, graph);

  // Finalize the function body.
  TF_EXPECT_OK(fbody.Finalize());

  // Check the function body properties after finalization.
  EXPECT_THAT(fbody.record, IsNull());
  EXPECT_THAT(fbody.graph, IsNull());
  EXPECT_THAT(fbody.arg_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.ret_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.arg_nodes, IsEmpty());
  EXPECT_THAT(fbody.ret_nodes, IsEmpty());
  EXPECT_THAT(fbody.control_ret_nodes, IsEmpty());
  EXPECT_THAT(
      fbody.args_alloc_attrs,
      UnorderedElementsAre(Property(&AllocatorAttributes::on_host, true)));
  EXPECT_THAT(
      fbody.rets_alloc_attrs,
      UnorderedElementsAre(Property(&AllocatorAttributes::on_host, true)));
}

TEST(FunctionBodyTest, FunctionBodyNotUpdatedWithFinalizationFailure) {
  core::RefCountPtr<FunctionRecord> record(
      new FunctionRecord(test::function::XTimesTwoWithControlOutput(),
                         /*stack_traces=*/{}, /*finalized=*/false));
  Graph* graph = new Graph(OpRegistry::Global());
  TF_ASSERT_OK(graph->AddNode(GetNodeDef(
      "x", FunctionLibraryDefinition::kArgOp, /*inputs=*/{},
      /*attrs=*/{{"T", DT_INT32}, {"index", 0}}, /*set_full_type_def=*/true)));
  TF_ASSERT_OK(
      graph->AddNode(GetNodeDef("y", FunctionLibraryDefinition::kRetOp,
                                /*inputs=*/{"output_const"},
                                /*attrs=*/{{"T", DT_INT32}, {"index", 0}})));
  TF_ASSERT_OK(graph->AddNode(GetNodeDef("dummy", "Const", /*inputs=*/{},
                                         /*attrs=*/{{"dtype", DT_INT32}})));
  FunctionBody fbody(std::move(record), {DT_INT32}, {DT_INT32}, graph);

  // Finalization fails due to missing input to the ret node.
  EXPECT_THAT(fbody.Finalize(),
              Not(absl_testing::StatusIs(absl::StatusCode::kOk)));

  // Check the function body properties after finalization.
  EXPECT_THAT(fbody.record, Not(IsNull()));
  EXPECT_THAT(fbody.graph, Not(IsNull()));
  EXPECT_THAT(fbody.arg_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.ret_types, UnorderedElementsAre(DT_INT32));
  EXPECT_THAT(fbody.arg_nodes,
              UnorderedElementsAre(Pointee(Property(
                  &Node::type_string, FunctionLibraryDefinition::kArgOp))));
  EXPECT_THAT(fbody.ret_nodes,
              UnorderedElementsAre(Pointee(Property(
                  &Node::type_string, FunctionLibraryDefinition::kRetOp))));
  EXPECT_THAT(fbody.control_ret_nodes,
              UnorderedElementsAre(Pointee(Property(&Node::name, "dummy"))));
  EXPECT_THAT(fbody.args_alloc_attrs, IsEmpty());
  EXPECT_THAT(fbody.rets_alloc_attrs, IsEmpty());
}

}  // namespace
}  // namespace tensorflow
