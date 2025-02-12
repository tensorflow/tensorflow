/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/inject_io_prefetch.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

using test::function::GDef;
using test::function::NDef;

FunctionDef InterleaveIoFunction(const std::string& name) {
  return FunctionDefHelper::Create(
      name,                   // function_name
      {"args_0: int64"},      // in_def
      {"identity: variant"},  // out_def
      {},                     // attr_def
      {
          // node_def
          {{"key_prefix"}, "Const", {}, {{"dtype", DT_STRING}}},
          {{"start_key"}, "Const", {}, {{"dtype", DT_STRING}}},
          {{"stop_key"}, "Const", {}, {{"dtype", DT_STRING}}},
          {{"SSTableDataset"},
           "SSTableDataset",
           {"args_0", "key_prefix:output:0", "start_key:output:0",
            "stop_key:output:0"},
           {}},
      },
      {});  // ret_def
}

GraphDef EligibleInterleaveCase() {
  return GDef(
      {NDef("files_string_1", "Const", {},
            {{"value", "file1file2"}, {"dtype", DT_STRING}}),
       NDef("files_tensor_1", "TensorSliceDataset", {"files_1_string"},
            {{"is_files", true}}),
       NDef("cycle_length_1", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length_1", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls_1", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelInterleaveV4Node(
           "interleave_1", "files_tensor_1", "cycle_length_1", "block_length_1",
           "num_parallel_calls_1", "io_1", /*deterministic=*/"default"),

       NDef("files_string_2", "Const", {},
            {{"value", "file1file2"}, {"dtype", DT_STRING}}),
       NDef("files_tensor_2", "TensorSliceDataset", {"files_2_string"},
            {{"is_files", true}}),
       NDef("cycle_length_2", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length_2", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls_2", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelInterleaveV4Node(
           "interleave_2", "files_tensor_2", "cycle_length_2", "block_length_2",
           "num_parallel_calls_2", "io_2", /*deterministic=*/"default"),

       NDef("zip", "ZipDataset", {"interleave_1", "interleave_2"}, {}),
       NDef("Sink", "Identity", {"zip"}, {})},

      {InterleaveIoFunction("io_1"), InterleaveIoFunction("io_2")});
}

GraphDef EligibleMapCase() {
  return GDef(
      {NDef("files_1", "Const", {},
            {{"value", "file1file2"}, {"dtype", DT_STRING}}),
       NDef("key_prefix_1", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("start_key_1", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("stop_key_1", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("io_1", "SSTableDataset",
            {"files_1", "key_prefix_1", "start_key_1", "stop_key_1"}, {}),
       NDef("num_parallel_calls_1", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapV2Node(
           "map_1", "io_1", "num_parallel_calls_1", "noop_1",
           /*deterministic=*/"default", /*use_unbounded_threadpool=*/false),

       NDef("files_2", "Const", {},
            {{"value", "file1file2"}, {"dtype", DT_STRING}}),
       NDef("key_prefix_2", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("start_key_2", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("stop_key_2", "Const", {}, {{"value", 1}, {"dtype", DT_STRING}}),
       NDef("io_2", "SSTableDataset",
            {"files_2", "key_prefix_2", "start_key_2", "stop_key_2"}, {}),
       NDef("num_parallel_calls_2", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapV2Node(
           "map_2", "io_2", "num_parallel_calls_2", "noop_2",
           /*deterministic=*/"default", /*use_unbounded_threadpool=*/false),

       NDef("zip", "ZipDataset", {"map_1", "map_2"}, {}),
       NDef("Sink", "Identity", {"zip"}, {})},

      {});
}

TEST(InjectIoPrefetchEligible, EligibleInterleaveCaseHasNoInjection) {
  GrapplerItem item;
  item.graph = EligibleInterleaveCase();
  item.fetch.push_back("Sink");

  InjectIoPrefetchEligible optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  NodeDef zip_node =
      output.node(graph_utils::FindGraphNodeWithName("zip", output));
  for (const auto& input_node_name : zip_node.input()) {
    NodeDef input_node = output.node(
        graph_utils::FindGraphNodeWithName(input_node_name, output));
    EXPECT_NE(input_node.op(), "PrefetchDataset");
  }
  EXPECT_EQ(item.graph.DebugString(), output.DebugString());
}

TEST(InjectIoPrefetchEligible, EligibleMapCaseHasNoInjection) {
  GrapplerItem item;
  item.graph = EligibleMapCase();
  item.fetch.push_back("Sink");

  InjectIoPrefetchEligible optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  NodeDef zip_node =
      output.node(graph_utils::FindGraphNodeWithName("zip", output));
  for (const auto& input_node_name : zip_node.input()) {
    NodeDef input_node = output.node(
        graph_utils::FindGraphNodeWithName(input_node_name, output));
    EXPECT_NE(input_node.op(), "PrefetchDataset");
  }
  EXPECT_EQ(item.graph.DebugString(), output.DebugString());
}

TEST(InjectIoPrefetch, InterleaveCaseHasInjection) {
  GrapplerItem item;
  item.graph = EligibleInterleaveCase();
  item.fetch.push_back("Sink");

  InjectIoPrefetch optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  NodeDef zip_node =
      output.node(graph_utils::FindGraphNodeWithName("zip", output));
  for (const auto& input_node_name : zip_node.input()) {
    NodeDef input_node = output.node(
        graph_utils::FindGraphNodeWithName(input_node_name, output));
    EXPECT_EQ(input_node.op(), "PrefetchDataset");
  }
}

TEST(InjectIoPrefetch, MapCaseHasInjection) {
  GrapplerItem item;
  item.graph = EligibleMapCase();
  item.fetch.push_back("Sink");

  InjectIoPrefetch optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  NodeDef zip_node =
      output.node(graph_utils::FindGraphNodeWithName("zip", output));
  for (const auto& input_node_name : zip_node.input()) {
    NodeDef input_node = output.node(
        graph_utils::FindGraphNodeWithName(input_node_name, output));
    EXPECT_EQ(input_node.op(), "PrefetchDataset");
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
