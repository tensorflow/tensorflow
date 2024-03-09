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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

// Implement a trivial version of the Rendezvous interface, to avoid
// clouding the benchmark results with the time spent in the various
// implementations, and to avoid the duplicate-send or duplicate-recv
// errors that would arise from running either benchmark in a loop.
class DummyRendezvous : public Rendezvous {
  Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
              const bool is_dead) override {
    return absl::OkStatus();
  }
  void RecvAsync(const ParsedKey& key, const Args& args,
                 DoneCallback done) override {
    static Tensor* t = new Tensor(DT_FLOAT, TensorShape({0}));
    done(absl::OkStatus(), args, args, *t, false);
  }
  void StartAbort(const Status& status) override {}
};

static Graph* Send() {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(DT_FLOAT, TensorShape({0}));
  test::graph::Send(g, test::graph::Constant(g, in0), "T", "/cpu:0", 1,
                    "/cpu:0");
  test::graph::Recv(g, "T", "float", "/cpu:0", 1, "/cpu:0");
  return g;
}

static Graph* Recv() {
  Graph* g = new Graph(OpRegistry::Global());
  test::graph::Recv(g, "T", "float", "/cpu:0", 1, "/cpu:0");
  return g;
}

void BM_Send(::testing::benchmark::State& state) {
  test::Benchmark("cpu", Send(), nullptr, nullptr, new DummyRendezvous, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_Send)->UseRealTime();

void BM_Recv(::testing::benchmark::State& state) {
  test::Benchmark("cpu", Recv(), nullptr, nullptr, new DummyRendezvous, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_Recv)->UseRealTime();

}  // namespace
}  // namespace tensorflow
