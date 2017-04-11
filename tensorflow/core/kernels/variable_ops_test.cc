/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

// Benchmark to simulate the overhead in training and serving workloads from too
// many threads grabbing the ResourceMgr lock at the same time because of the
// variable and queue ops.
void ManyManyVariablesHelper(int threads, int variables, int iters) {
  testing::StopTiming();
  Graph g(OpRegistry::Global());
  std::vector<string> targets;
  for (int i = 0; i < variables; ++i) {
    Node* v;
    TF_CHECK_OK(
        NodeBuilder(
            g.NewName("VeryVeryLongRealistSoundingVariableName/weights"),
            "VariableV2")
            .Attr("shape", TensorShape())
            .Attr("dtype", DT_FLOAT)
            .Finalize(&g, &v));
    targets.push_back(v->name());
  }
  GraphDef gd;
  g.ToGraphDef(&gd);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(threads);
  Session* sess = NewSession(opts);
  TF_CHECK_OK(sess->Create(gd));
  TF_CHECK_OK(sess->Run({}, {}, targets, nullptr));
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(sess->Run({}, {}, targets, nullptr));
  }
  testing::StopTiming();
  delete sess;
}

void BM_ManyManyVariablesManyThreads(int iters, int threads) {
  ManyManyVariablesHelper(threads, 1000, iters);
}

BENCHMARK(BM_ManyManyVariablesManyThreads)->Arg(50);

}  // namespace
}  // namespace tensorflow
