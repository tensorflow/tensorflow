/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/strings/proto_serialization.h"

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
GraphDef MakeGraphDef(int num_nodes) {
  GraphDef graph_def;
  for (int i = 0; i < num_nodes; ++i) {
    NodeDef* node = graph_def.add_node();
    node->set_name(strings::StrCat("node", i));
    node->set_op(strings::StrCat("op", i % 10));
    (*node->mutable_attr())["foo"].set_f(3.14f);
    (*node->mutable_attr())["bar"].set_s("baz");
  }
  return graph_def;
}
}  // namespace

static void BM_ProtoSerializationToString(::testing::benchmark::State& state) {
  int num_nodes = state.range(0);

  GraphDef graph_def = MakeGraphDef(num_nodes);

  for (auto i : state) {
    string serialized;
    testing::DoNotOptimize(
        SerializeToStringDeterministic(graph_def, &serialized));
  }
}

BENCHMARK(BM_ProtoSerializationToString)->Range(1, 10000);

static void BM_ProtoSerializationToBuffer(::testing::benchmark::State& state) {
  int num_nodes = state.range(0);

  GraphDef graph_def = MakeGraphDef(num_nodes);

  const size_t size = graph_def.ByteSizeLong();
  for (auto i : state) {
    absl::InlinedVector<char, 1024UL> buf(size);
    testing::DoNotOptimize(
        SerializeToBufferDeterministic(graph_def, buf.data(), size));
  }
}
BENCHMARK(BM_ProtoSerializationToBuffer)->Range(1, 10000);

static void BM_DeterministicProtoHash64(::testing::benchmark::State& state) {
  int num_nodes = state.range(0);

  GraphDef graph_def = MakeGraphDef(num_nodes);

  for (auto i : state) {
    testing::DoNotOptimize(DeterministicProtoHash64(graph_def));
  }
}
BENCHMARK(BM_DeterministicProtoHash64)->Range(1, 10000);

static void BM_AreSerializedProtosEqual(::testing::benchmark::State& state) {
  int num_nodes = state.range(0);

  GraphDef graph_def_a = MakeGraphDef(num_nodes);
  GraphDef graph_def_b = MakeGraphDef(num_nodes);
  graph_def_b.mutable_node(0)->mutable_name()[0] = 'l';

  for (auto i : state) {
    testing::DoNotOptimize(AreSerializedProtosEqual(graph_def_a, graph_def_a));
  }
}
BENCHMARK(BM_AreSerializedProtosEqual)->Range(1, 10000);

}  // namespace tensorflow
