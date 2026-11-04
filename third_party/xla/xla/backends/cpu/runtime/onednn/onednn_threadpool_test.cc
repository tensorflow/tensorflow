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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "xla/backends/cpu/runtime/onednn/onednn_interop.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
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

#ifdef ENABLE_ONEDNN_ASYNC
  OneDnnThreadPool threadpool(threads.AsEigenThreadPool(), /*is_async=*/true);
#else
  OneDnnThreadPool threadpool(threads.AsEigenThreadPool());
#endif  // ENABLE_ONEDNN_ASYNC

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

#ifdef ENABLE_ONEDNN_ASYNC
  stream.wait();
#endif

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_NEAR(dst_data[i], std::exp(1.0f), 1e-5);
  }
}

TEST(OneDnnThreadPoolTest, FallbackThreadPool) {
  Eigen::ThreadPoolInterface* fallback_pool = GetFallbackThreadPoolForOneDnn();
  ASSERT_NE(fallback_pool, nullptr);
  EXPECT_EQ(fallback_pool->NumThreads(), 1);

  OneDnnThreadPool threadpool(fallback_pool);

  int64_t d0 = 4;
  int64_t d1 = 8;
  int64_t num_elements = d0 * d1;

  dnnl::graph::logical_tensor::dims dims = {d0, d1};
  dnnl::graph::logical_tensor::dims strides = {d1, 1};

  dnnl::graph::logical_tensor src_tensor(
      0, dnnl::graph::logical_tensor::data_type::f32, dims, strides);
  dnnl::graph::logical_tensor dst_tensor(
      1, dnnl::graph::logical_tensor::data_type::f32, dims, strides);

  TF_ASSERT_OK_AND_ASSIGN(dnnl::graph::graph g,
                          CreateExpGraph(src_tensor, dst_tensor));
  std::vector<dnnl::graph::partition> partitions = g.get_partitions();

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, &threadpool));

  std::vector<dnnl::graph::compiled_partition> compiled_partitions;
  for (const auto& partition : partitions) {
    compiled_partitions.push_back(
        partition.compile({src_tensor}, {dst_tensor}, engine));
  }

  std::vector<float> src_data(num_elements, 1.0f);
  std::vector<float> dst_data(num_elements, 0.0f);

  dnnl::graph::tensor src(src_tensor, engine, src_data.data());
  dnnl::graph::tensor dst(dst_tensor, engine, dst_data.data());

  compiled_partitions[0].execute(stream, {src}, {dst});

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_NEAR(dst_data[i], std::exp(1.0f), 1e-5);
  }
}

TEST(OneDnnThreadPoolTest, GetInParallelTracking) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);

  for (bool is_async : {false, true}) {
    OneDnnThreadPool threadpool(threads.AsEigenThreadPool(), is_async);

    // Outside parallel_for on the calling thread, get_in_parallel() is false.
    EXPECT_FALSE(threadpool.get_in_parallel());

    // Outside parallel_for on an Eigen worker thread (simulating ThunkExecutor
    // dispatching a thunk onto intra_op_threadpool), get_in_parallel() must
    // still be false so top-level oneDNN primitives run multi-threaded.
    absl::Notification worker_done;
    threads.Schedule([&] {
      EXPECT_FALSE(threadpool.get_in_parallel());
      worker_done.Notify();
    });
    worker_done.WaitForNotification();

    // Inside parallel_for, get_in_parallel() must be true on every worker.
    constexpr int kNumTasks = 16;
    std::atomic<int> in_parallel_true_count{0};
    threadpool.parallel_for(kNumTasks, [&](int i, int n) {
      EXPECT_EQ(n, kNumTasks);
      if (threadpool.get_in_parallel()) {
        in_parallel_true_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
    if (is_async) {
      tsl::BlockUntilReady(threadpool.done_event());
    }
    EXPECT_EQ(in_parallel_true_count.load(), kNumTasks);
    EXPECT_FALSE(threadpool.get_in_parallel());
  }
}

TEST(OneDnnThreadPoolTest, NestedParallelForAsyncRaceAndCompletion) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  OneDnnThreadPool threadpool(threads.AsEigenThreadPool(), /*is_async=*/true);

  constexpr int kOuterTasks = 32;
  constexpr int kInnerTasks = 16;
  std::atomic<int> completed_inner_tasks{0};

  threadpool.parallel_for(kOuterTasks, [&](int outer_i, int outer_n) {
    EXPECT_EQ(outer_n, kOuterTasks);
    EXPECT_TRUE(threadpool.get_in_parallel());

    // Directly invoking nested parallel_for from multiple outer workers must
    // not race on `done_event_` and must finish before the outer done_event
    // completes.
    threadpool.parallel_for(kInnerTasks, [&](int inner_i, int inner_n) {
      EXPECT_EQ(inner_n, kInnerTasks);
      EXPECT_TRUE(threadpool.get_in_parallel());
      completed_inner_tasks.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_TRUE(threadpool.get_in_parallel());
  });

  // Capture done_event immediately after outer parallel_for returns (matching
  // OneDnnFusionThunk::OneDnnRuntime::Invoke).
  tsl::AsyncValueRef<tsl::Chain> done = threadpool.done_event();
  tsl::BlockUntilReady(done);

  EXPECT_EQ(completed_inner_tasks.load(std::memory_order_relaxed),
            kOuterTasks * kInnerTasks);
  EXPECT_FALSE(threadpool.get_in_parallel());
}

}  // namespace
}  // namespace xla::cpu
