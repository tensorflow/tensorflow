/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class BenchmarkKernel : public OpKernel {
 public:
  explicit BenchmarkKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

// Macro to register a fake op and kernel.
// The average number of attrs across all TF ops is 3.
#define REGISTER_BENCHMARK_OP(N)                          \
  REGISTER_OP("BenchmarkOp" #N)                           \
      .Input("a: T")                                      \
      .Output("b: T")                                     \
      .Attr("T: {float}")                                 \
      .Attr("attr_3: int = 42")                           \
      .Attr("attr_0: int = 0")                            \
      .Attr("attr_1: int = 1")                            \
      .Attr("attr_2: int = 2");                           \
  REGISTER_KERNEL_BUILDER(Name("BenchmarkOp" #N)          \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<float>("T") \
                              .Priority(1),               \
                          BenchmarkKernel)

// Recursive macros to generate many registrations.
#define REPEAT_1(M, X) M(X)
#define REPEAT_2(M, X) M(X##0) M(X##1)
#define REPEAT_4(M, X) REPEAT_2(M, X##0) REPEAT_2(M, X##1)
#define REPEAT_8(M, X) REPEAT_4(M, X##0) REPEAT_4(M, X##1)
#define REPEAT_16(M, X) REPEAT_8(M, X##0) REPEAT_8(M, X##1)
#define REPEAT_32(M, X) REPEAT_16(M, X##0) REPEAT_16(M, X##1)
#define REPEAT_64(M, X) REPEAT_32(M, X##0) REPEAT_32(M, X##1)
#define REPEAT_128(M, X) REPEAT_64(M, X##0) REPEAT_64(M, X##1)
#define REPEAT_256(M, X) REPEAT_128(M, X##0) REPEAT_128(M, X##1)
#define REPEAT_512(M, X) REPEAT_256(M, X##0) REPEAT_256(M, X##1)
#define REPEAT_1024(M, X) REPEAT_512(M, X##0) REPEAT_512(M, X##1)

// Register 1024 kernels.
REPEAT_1024(REGISTER_BENCHMARK_OP, 0)

void BM_FindKernelDef(benchmark::State& state) {
  // Pre-compute NodeDefs with random op names to avoid benchmark overhead.
  // We use a relatively large number of NodeDefs to avoid cache effects,
  // but small enough to fit in memory.
  const int kNumNodeDefs = 1024 * 10;

  // Generate a random set of node defs to lookup.
  static const std::vector<NodeDef>* node_defs = [] {
    auto* defs = new std::vector<NodeDef>();
    defs->reserve(kNumNodeDefs);
    std::mt19937 gen(42);  // Fixed seed for reproducibility.
    std::uniform_int_distribution<> dist(0, kNumNodeDefs - 1);
    for (int i = 0; i < kNumNodeDefs; ++i) {
      int random_index = dist(gen);
      std::string suffix = "0";
      for (int bit = 9; bit >= 0; --bit) {
        suffix += ((random_index >> bit) & 1) ? "1" : "0";
      }
      std::string op_name = "BenchmarkOp" + suffix;

      NodeDef node_def;
      TF_CHECK_OK(NodeDefBuilder("benchmark-op", op_name)
                      .Input("a", 0, DT_FLOAT)
                      .Attr("attr_3", 42)
                      .Attr("attr_0", 0)
                      .Attr("attr_1", 1)
                      .Attr("attr_2", 2)
                      .Finalize(&node_def));
      defs->push_back(std::move(node_def));
    }
    return defs;
  }();

  const DeviceType device_type(DEVICE_CPU);
  const KernelDef* def = nullptr;
  std::string kernel_class_name;

  int64_t i(0);
  for (auto s : state) {
    const NodeDef& node_def = (*node_defs)[i % kNumNodeDefs];
    absl::Status status =
        FindKernelDef(device_type, node_def, &def, &kernel_class_name);
    if (!status.ok()) {
      state.SkipWithError(status.ToString());
      break;
    }
    i++;
  }
}

void BM_FindKernelDef_Arguments(::benchmark::internal::Benchmark* b) {
  int num_cpus = port::NumSchedulableCPUs();
  if (num_cpus <= 0) num_cpus = 16;  // Fallback.
  b->ThreadRange(1, num_cpus);
}
BENCHMARK(BM_FindKernelDef)->Apply(BM_FindKernelDef_Arguments);

}  // namespace
}  // namespace tensorflow
