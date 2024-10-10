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

#include "tensorflow/core/grappler/optimizers/data/seq_interleave_prefetch.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace grappler {
namespace {

using test::function::GDef;
using test::function::NDef;

constexpr char kPrefetchDatasetOpName[] = "PrefetchDataset";
constexpr char kInterleaveDatasetOpName[] = "InterleaveDataset";
constexpr char kParallelInterleaveDatasetOpName[] =
    "ParallelInterleaveDatasetV4";
constexpr char kSeqInterleavePrefetchRewritePrefix[] =
    "inject/seq_interleave_prefetch_rewrite_";
constexpr char kFdefProtoStr[] =
    R"pb(signature {
           name: "parallel_interleave_fdef"
           input_arg { name: "args_0" type: DT_STRING }
           output_arg { name: "identity" type: DT_VARIANT }
           is_stateful: true
           control_output: "SSTableDataset"
         }
         node_def {
           name: "key_prefix"
           op: "Const"
           attr {
             key: "dtype"
             value { type: DT_STRING }
           }
           attr {
             key: "value"
             value {
               tensor {
                 dtype: DT_STRING
                 tensor_shape {}
                 string_val: ""
               }
             }
           }
         }
         node_def {
           name: "start_key"
           op: "Const"
           attr {
             key: "dtype"
             value { type: DT_STRING }
           }
           attr {
             key: "value"
             value {
               tensor {
                 dtype: DT_STRING
                 tensor_shape {}
                 string_val: ""
               }
             }
           }
         }
         node_def {
           name: "stop_key"
           op: "Const"
           attr {
             key: "dtype"
             value { type: DT_STRING }
           }
           attr {
             key: "value"
             value {
               tensor {
                 dtype: DT_STRING
                 tensor_shape {}
                 string_val: ""
               }
             }
           }
         }
         node_def {
           name: "SSTableDataset"
           op: "SSTableDataset"
           input: "args_0"
           input: "key_prefix:output:0"
           input: "start_key:output:0"
           input: "stop_key:output:0"
           attr {
             key: "metadata"
             value { s: "" }
           }
           attr {
             key: "split_size"
             value { i: 0 }
           }
           experimental_type {
             type_id: TFT_PRODUCT
             args {
               type_id: TFT_DATASET
               args {
                 type_id: TFT_TENSOR
                 args { type_id: TFT_STRING }
               }
             }
           }
         }
         node_def {
           name: "Identity"
           op: "Identity"
           input: "SSTableDataset:handle:0"
           input: "^NoOp"
           attr {
             key: "T"
             value { type: DT_VARIANT }
           }
         }
         node_def { name: "NoOp" op: "NoOp" input: "^SSTableDataset" }
         ret { key: "identity" value: "Identity:output:0" }
         attr {
           key: "_construction_context"
           value { s: "kEagerRuntime" }
         }
         attr {
           key: "_tf_data_function"
           value { b: true }
         }
         control_ret { key: "SSTableDataset" value: "SSTableDataset" }
         arg_attr {
           key: 0
           value {
             attr {
               key: "_output_shapes"
               value { list { shape {} } }
             }
             attr {
               key: "_user_specified_name"
               value { s: "args_0" }
             }
           }
         })pb";

GraphDef ParallelInterleaveCase(bool deterministic) {
  FunctionDef fdef;
  protobuf::TextFormat::ParseFromString(kFdefProtoStr, &fdef);
  return GDef(
      {NDef("stop", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"stop"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelInterleaveV4Node(
           "parallel_interleave", "range", "cycle_length", "block_length",
           "num_parallel_calls", "parallel_interleave_fdef",
           deterministic ? "true" : "false")},
      // FunctionLib
      {
          fdef,
      });
}

GraphDef MultipleParallelInterleaveCase(bool deterministic) {
  FunctionDef fdef_1, fdef_2, fdef_3;
  protobuf::TextFormat::ParseFromString(kFdefProtoStr, &fdef_1);
  fdef_1.mutable_signature()->set_name("parallel_interleave_fdef_1");
  protobuf::TextFormat::ParseFromString(kFdefProtoStr, &fdef_2);
  fdef_2.mutable_signature()->set_name("parallel_interleave_fdef_2");
  protobuf::TextFormat::ParseFromString(kFdefProtoStr, &fdef_3);
  fdef_3.mutable_signature()->set_name("parallel_interleave_fdef_3");

  auto make_parallel_interleave_node =
      [&deterministic](const int node_num, const FunctionDef &fdef) {
        return graph_tests_utils::MakeParallelInterleaveV4Node(
            absl::StrCat("parallel_interleave_", node_num), "range",
            "cycle_length", "block_length", "num_parallel_calls",
            fdef.signature().name(), deterministic ? "true" : "false");
      };

  return GDef(
      {NDef("stop", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"stop"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       make_parallel_interleave_node(1, fdef_1),
       make_parallel_interleave_node(2, fdef_2),
       make_parallel_interleave_node(3, fdef_3)},
      // FunctionLib
      {
          fdef_1,
          fdef_2,
          fdef_3,
      });
}

GraphDef InterleaveCase(bool deterministic) {
  FunctionDef fdef;
  protobuf::TextFormat::ParseFromString(kFdefProtoStr, &fdef);
  return GDef(
      {NDef("stop", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"stop"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeInterleaveNode(
           "sequential_interleave", "range", "cycle_length", "block_length",
           "parallel_interleave_fdef", deterministic ? "true" : "false")},
      // FunctionLib
      {
          fdef,
      });
}

bool PrefetchInFunction(const NodeDef &node,
                        const FunctionLibraryDefinition &flib) {
  auto f_attr_it = node.attr().find("f");
  if (f_attr_it == node.attr().end()) return false;
  const FunctionDef *func = flib.Find(f_attr_it->second.func().name());
  if (func == nullptr) {
    return false;
  }
  for (int i = 0; i < func->node_def_size(); i++) {
    NodeDef node_in_func = func->node_def(i);
    if (tensorflow::data::MatchesAnyVersion(
            /*op_prefix=*/kPrefetchDatasetOpName,
            /*op_to_match=*/node_in_func.op())) {
      return true;
    }
  }
  return false;
}

bool IsInterleaveNode(const NodeDef &node) {
  return (node.op() == kInterleaveDatasetOpName);
}

}  // namespace

absl::Status OptimizeWithInjectInterleavePrefetch(const GrapplerItem &item,
                                                  GraphDef *output) {
  SeqInterleavePrefetch optimizer;
  return optimizer.Optimize(nullptr, item, output);
}

class SeqInterleavePrefetchParameterizedTest
    : public ::testing::TestWithParam<bool> {};

TEST_P(SeqInterleavePrefetchParameterizedTest,
       ParallelInterleaveHasConditionalInjection) {
  GrapplerItem item;
  bool deterministic = GetParam();
  item.graph = ParallelInterleaveCase(deterministic);
  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectInterleavePrefetch(item, &output));
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), output.library());
  const std::string &parallel_interleave_fdef_name = "parallel_interleave_fdef";
  const std::string &interleave_fdef_name = absl::StrCat(
      kSeqInterleavePrefetchRewritePrefix, parallel_interleave_fdef_name);
  if (deterministic) {
    EXPECT_TRUE(
        !graph_utils::ContainsGraphNodeWithName("parallel_interleave", output));
    EXPECT_TRUE(!graph_utils::ContainsNodeWithOp(
        kParallelInterleaveDatasetOpName, output));
    EXPECT_TRUE(
        graph_utils::ContainsNodeWithOp(kInterleaveDatasetOpName, output));
    for (auto node : output.node()) {
      if (!IsInterleaveNode(node)) continue;
      EXPECT_TRUE(PrefetchInFunction(node, lib_def));
    }
    const FunctionDef *parallel_interleave_fdef =
        lib_def.Find(parallel_interleave_fdef_name);
    const FunctionDef *interleave_fdef = lib_def.Find(interleave_fdef_name);
    EXPECT_EQ(parallel_interleave_fdef, nullptr);
    EXPECT_NE(interleave_fdef, nullptr);
    EXPECT_EQ(lib_def.ListFunctionNames().at(0), interleave_fdef_name);
    EXPECT_TRUE(function_utils::FindFunctionNodeWithOp(kPrefetchDatasetOpName,
                                                       *interleave_fdef));
  } else {
    EXPECT_TRUE(graph_utils::ContainsNodeWithOp(
        kParallelInterleaveDatasetOpName, output));
    EXPECT_TRUE(
        !graph_utils::ContainsNodeWithOp(kInterleaveDatasetOpName, output));
    EXPECT_TRUE(
        graph_utils::ContainsGraphNodeWithName("parallel_interleave", output));
    EXPECT_NE(lib_def.Find(parallel_interleave_fdef_name), nullptr);
  }
  EXPECT_EQ(lib_def.num_functions(), 1);
}

TEST_P(SeqInterleavePrefetchParameterizedTest,
       MultipleParallelInterleavesHaveConditionalInjection) {
  GrapplerItem item;
  bool deterministic = GetParam();
  item.graph = MultipleParallelInterleaveCase(deterministic);
  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectInterleavePrefetch(item, &output));
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), output.library());
  if (deterministic) {
    EXPECT_TRUE(!graph_utils::ContainsNodeWithOp(
        kParallelInterleaveDatasetOpName, output));
    EXPECT_TRUE(
        graph_utils::ContainsNodeWithOp(kInterleaveDatasetOpName, output));
    for (int i = 1; i <= 3; ++i) {
      EXPECT_TRUE(!graph_utils::ContainsGraphNodeWithName(
          absl::StrCat("parallel_interleave_", std::to_string(i)), output));
    }
    for (auto node : output.node()) {
      if (!IsInterleaveNode(node)) continue;
      EXPECT_TRUE(PrefetchInFunction(node, lib_def));
    }
  } else {
    EXPECT_TRUE(graph_utils::ContainsNodeWithOp(
        kParallelInterleaveDatasetOpName, output));
    EXPECT_TRUE(
        !graph_utils::ContainsNodeWithOp(kInterleaveDatasetOpName, output));
    for (int i = 1; i <= 3; ++i) {
      EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName(
          absl::StrCat("parallel_interleave_", std::to_string(i)), output));
    }
  }
}

TEST_P(SeqInterleavePrefetchParameterizedTest,
       SequentialInterleaveHasNoInjection) {
  GrapplerItem item;
  item.graph = InterleaveCase(/*deterministic=*/GetParam());
  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectInterleavePrefetch(item, &output));
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp(kInterleaveDatasetOpName, output));
  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName("sequential_interleave", output));
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), output.library());
  for (auto node : output.node()) {
    if (!IsInterleaveNode(node)) continue;
    EXPECT_FALSE(PrefetchInFunction(node, lib_def));
  }
}

INSTANTIATE_TEST_SUITE_P(Determinism, SeqInterleavePrefetchParameterizedTest,
                         ::testing::Values(false, true));

}  // namespace grappler
}  // namespace tensorflow
