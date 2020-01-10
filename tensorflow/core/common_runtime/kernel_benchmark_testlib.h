#ifndef TENSORFLOW_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_
#define TENSORFLOW_COMMON_RUNTIME_KERNEL_BENCHMARK_TESTLIB_H_

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class Device;
class SessionOptions;

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
