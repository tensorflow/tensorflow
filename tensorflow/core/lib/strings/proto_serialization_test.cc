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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static void BM_ProtoSerializationToString(int iters, int num_nodes) {
  testing::StopTiming();
  GraphDef graph_def;
  for (int i = 0; i < num_nodes; ++i) {
    NodeDef* node = graph_def.add_node();
    node->set_name(strings::StrCat("node", i));
    node->set_op(strings::StrCat("op", i % 10));
  }
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    string serialized;
    testing::DoNotOptimize(
        SerializeToStringDeterministic(graph_def, &serialized));
  }
  testing::StopTiming();
}
BENCHMARK(BM_ProtoSerializationToString)->Range(1, 10000);

static void BM_ProtoSerializationToBuffer(int iters, int num_nodes) {
  testing::StopTiming();
  GraphDef graph_def;
  for (int i = 0; i < num_nodes; ++i) {
    NodeDef* node = graph_def.add_node();
    node->set_name(strings::StrCat("node", i));
    node->set_op(strings::StrCat("op", i % 10));
  }
  testing::StartTiming();
  const size_t size = graph_def.ByteSizeLong();
  for (int i = 0; i < iters; ++i) {
    auto buf = absl::make_unique<char[]>(size);
    testing::DoNotOptimize(
        SerializeToBufferDeterministic(graph_def, buf.get(), size));
  }
  testing::StopTiming();
}

BENCHMARK(BM_ProtoSerializationToBuffer)->Range(1, 10000);

}  // namespace tensorflow
