/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/onednn/onednn_interop.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

// Creates a graph with a single Exp operation.
static absl::StatusOr<dnnl::graph::graph> CreateExpGraph(
    const dnnl::graph::logical_tensor& src_tensor,
    const dnnl::graph::logical_tensor& dst_tensor) {
  dnnl::graph::op exp_op(0, dnnl::graph::op::kind::Exp, {src_tensor},
                         {dst_tensor});

  dnnl::graph::graph g(dnnl::engine::kind::cpu);
  ONEDNN_RETURN_IF_ERROR(g.add_op(exp_op));
  g.finalize();

  return g;
}

TEST(OneDnnThreadPoolTest, Binary) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 32);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  ParallelLoopRunner runner(&device);
  OneDnnThreadPool threadpool(&runner);

  int64_t d0 = 100;
  int64_t d1 = 1000;
  int64_t num_elements = d0 * d1;

  // We use row-major layout for both source and destination tensors.
  dnnl::graph::logical_tensor::dims src_dims = {d0, d1};
  dnnl::graph::logical_tensor::dims dst_dims = {d0, d1};

  dnnl::graph::logical_tensor::dims src_strides = {d1, 1};
  dnnl::graph::logical_tensor::dims dst_strides = {d1, 1};

  dnnl::graph::logical_tensor src_tensor(
      0, dnnl::graph::logical_tensor::data_type::f32, src_dims, src_strides);
  dnnl::graph::logical_tensor dst_tensor(
      1, dnnl::graph::logical_tensor::data_type::f32, dst_dims, dst_strides);

  // Compile oneDNN graph with a single Exp operation.
  TF_ASSERT_OK_AND_ASSIGN(dnnl::graph::graph g,
                          CreateExpGraph(src_tensor, dst_tensor));
  std::vector<dnnl::graph::partition> partitions = g.get_partitions();

  // Create oneDNN engine for running the graph on CPU.
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);

  // Create oneDNN stream backed by parallel loop runner.
  dnnl::stream stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, &threadpool));

  // Compile graph partitions for given engine.
  std::vector<dnnl::graph::compiled_partition> compiled_partitions;
  for (const auto& partition : partitions) {
    compiled_partitions.push_back(
        partition.compile({src_tensor}, {dst_tensor}, engine));
  }

  // Create tensors for source and destination.
  std::vector<float> src_data(num_elements, 1.0f);
  std::vector<float> dst_data(num_elements, 0.0f);

  dnnl::graph::tensor src(src_tensor, engine, src_data.data());
  dnnl::graph::tensor dst(dst_tensor, engine, dst_data.data());

  // Execute compiled oneDNN graph on the CPU stream.
  compiled_partitions[0].execute(stream, {src}, {dst});

  // Wait for the completion of parallel loops scheduled into the runner.
  tsl::BlockUntilReady(runner.done_event());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_NEAR(dst_data[i], std::exp(1.0f), 1e-5);
  }
}

}  // namespace
}  // namespace xla::cpu
