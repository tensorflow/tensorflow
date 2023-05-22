/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
class FunctionLibraryRuntime;
class ProcessFunctionLibraryRuntime;
struct SessionOptions;
class DynamicDeviceMgr;

namespace test {

class Benchmark {
 public:
  // "device" must be either "cpu" or "gpu".  Takes ownership of "g",
  // "init", and one reference on "rendez" (if not null).
  //
  // old_benchmark_api: If true, the benchmark is running with older API
  //   * In the old API, the timer needs to be stopped/restarted
  //     by users.
  //   * In the new API, the timer starts automatically at the first
  //     iteration of the loop and stops after the last iteration.
  // TODO(vyng) Remove this once we have migrated all code to newer API.
  Benchmark(const string& device, Graph* g,
            const SessionOptions* options = nullptr, Graph* init = nullptr,
            Rendezvous* rendez = nullptr, const char* executor_type = "",
            bool old_benchmark_api = false);

  Benchmark(const string& device, Graph* g, bool old_benchmark_api);

  ~Benchmark();

  void Run(benchmark::State& state);

  void RunWithRendezvousArgs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      const std::vector<string>& outputs, benchmark::State& state);

 private:
  thread::ThreadPool* pool_ = nullptr;  // Not owned.
  Device* device_ = nullptr;            // Not owned.
  Rendezvous* rendez_ = nullptr;
  std::unique_ptr<DynamicDeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* flr_;  // Not owned.
  std::unique_ptr<Executor> exec_;

  TF_DISALLOW_COPY_AND_ASSIGN(Benchmark);
};

// Returns the rendezvous key associated with the given Send/Recv node.
string GetRendezvousKey(const Node* node);

}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_
