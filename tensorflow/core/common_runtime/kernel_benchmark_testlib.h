/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_
#define TENSORFLOW_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
struct SessionOptions;

namespace test {

class Benchmark {
 public:
  // "device" must be either "cpu" or "gpu".  Takes ownership of "g"
  // and "init".
  Benchmark(const string& device, Graph* g,
            const SessionOptions* options = nullptr, Graph* init = nullptr);
  ~Benchmark();

  // Executes the graph for "iters" times.
  void Run(int iters);

  // If "g" contains send/recv nodes, before each execution, we send
  // inputs to the corresponding recv nodes in the graph, after each
  // execution, we recv outputs from the corresponding send nodes in
  // the graph. In the benchmark, we throw away values returned by the
  // graph.
  void RunWithArgs(const std::vector<std::pair<const Node*, Tensor>>& inputs,
                   const std::vector<const Node*>& outputs, int iters);

 private:
  thread::ThreadPool* pool_ = nullptr;
  thread::ThreadPool* non_blocking_pool_ = nullptr;
  Device* device_ = nullptr;
  Rendezvous* rendez_ = nullptr;
  Executor* exec_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(Benchmark);
};

}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_
