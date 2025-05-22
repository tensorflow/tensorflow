/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/xnn_threadpool.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "xnnpack.h"
#include "absl/algorithm/container.h"
#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

static xnn_status CreateBinaryOpsSubgraph(xnn_subgraph_t subgraph,
                                          std::vector<size_t> dims) {
  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out0_id = XNN_INVALID_VALUE_ID;
  uint32_t out1_id = XNN_INVALID_VALUE_ID;

  if (auto s = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                       dims.data(), nullptr, /*external_id=*/0,
                                       XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                       dims.data(), nullptr, /*external_id=*/1,
                                       XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr,
          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out0_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr,
          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out1_id);
      s != xnn_status_success) {
    return s;
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};

  if (auto s = xnn_define_binary(subgraph, xnn_binary_add, &params, lhs_id,
                                 rhs_id, out0_id, /*flags=*/0);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_binary(subgraph, xnn_binary_multiply, &params, lhs_id,
                                 rhs_id, out1_id, /*flags=*/0);
      s != xnn_status_success) {
    return s;
  }

  return xnn_status_success;
}

static xnn_status CreateDotSubgraph(xnn_subgraph_t subgraph, size_t m, size_t n,
                                    size_t k) {
  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  std::vector<size_t> lhs_dims = {m, k};
  std::vector<size_t> rhs_dims = {k, n};
  std::vector<size_t> out_dims = {m, n};

  if (auto s = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, lhs_dims.size(), lhs_dims.data(),
          nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, rhs_dims.size(), rhs_dims.data(),
          nullptr, /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(),
          nullptr,
          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);
      s != xnn_status_success) {
    return s;
  }

  if (auto s =
          xnn_define_batch_matrix_multiply(subgraph, lhs_id, rhs_id, out_id,
                                           /*flags=*/0);
      s != xnn_status_success) {
    return s;
  }

  return xnn_status_success;
}

class XnnThreadPoolTest : public testing::TestWithParam<bool> {
 public:
  XnnThreadPoolTest()
      : thread_pool_(tsl::Env::Default(), "xnn-threadpool-test", 8),
        device_(thread_pool_.AsEigenThreadPool(), thread_pool_.NumThreads()),
        runner_(&device_) {}

  pthreadpool_t CreateThreadPool() {
    return GetParam() ? pthreadpool_create(8)
                      : CreateCustomPthreadpool(&runner_);
  }

  void DestroyThreadPool(pthreadpool_t threadpool) {
    if (GetParam()) {
      pthreadpool_destroy(threadpool);
    } else {
      DestroyCustomPthreadpool(threadpool);
    }
  }

 private:
  tsl::thread::ThreadPool thread_pool_;
  Eigen::ThreadPoolDevice device_;
  ParallelLoopRunner runner_;
};

TEST_P(XnnThreadPoolTest, Binary) {
  pthreadpool_t threadpool = CreateThreadPool();
  ASSERT_NE(threadpool, nullptr);

  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);

  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_create_workspace(&workspace), xnn_status_success);

  xnn_subgraph_t subgraph = nullptr;

  ASSERT_EQ(
      xnn_create_subgraph(/*external_value_ids=*/4, /*flags=*/0, &subgraph),
      xnn_status_success);

  size_t d0 = 1024;
  CreateBinaryOpsSubgraph(subgraph, {d0, d0});

  std::vector<float> lhs(d0 * d0, 2.0f);
  std::vector<float> rhs(d0 * d0, 3.0f);
  std::vector<float> out0(d0 * d0, 0.0f);
  std::vector<float> out1(d0 * d0, 0.0f);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_create_runtime_v4(subgraph, nullptr, workspace, threadpool, 0,
                                  &runtime),
            xnn_status_success);

  std::vector<xnn_external_value> external_values = {
      xnn_external_value{0, lhs.data()},
      xnn_external_value{1, rhs.data()},
      xnn_external_value{2, out0.data()},
      xnn_external_value{3, out1.data()},
  };

  ASSERT_EQ(xnn_reshape_runtime(runtime), xnn_status_success);
  ASSERT_EQ(xnn_setup_runtime_v2(runtime, 4, external_values.data()),
            xnn_status_success);

  ASSERT_EQ(xnn_invoke_runtime(runtime), xnn_status_success);

  if (ParallelLoopRunner* runner = GetParallelLoopRunner(threadpool)) {
    tsl::BlockUntilReady(runner->done_event());
    ASSERT_TRUE(runner->done_event().IsConcrete());
  }

  ASSERT_TRUE(absl::c_all_of(out0, [](float v) { return v == 5.0f; }));
  ASSERT_TRUE(absl::c_all_of(out1, [](float v) { return v == 6.0f; }));

  ASSERT_EQ(xnn_delete_runtime(runtime), xnn_status_success);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
  ASSERT_EQ(xnn_release_workspace(workspace), xnn_status_success);

  DestroyThreadPool(threadpool);
}

TEST_P(XnnThreadPoolTest, Dot) {
  pthreadpool_t threadpool = CreateThreadPool();
  ASSERT_NE(threadpool, nullptr);

  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);

  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_create_workspace(&workspace), xnn_status_success);

  xnn_subgraph_t subgraph = nullptr;

  ASSERT_EQ(
      xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph),
      xnn_status_success);

  size_t m = 256, k = 256, n = 256;
  CreateDotSubgraph(subgraph, m, k, n);

  std::vector<float> lhs(m * k, 1.0f);
  std::vector<float> rhs(k * n, 1.0f);
  std::vector<float> out(m * n, 0.0f);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_create_runtime_v4(subgraph, nullptr, workspace, threadpool, 0,
                                  &runtime),
            xnn_status_success);

  std::vector<xnn_external_value> external_values = {
      xnn_external_value{0, lhs.data()},
      xnn_external_value{1, rhs.data()},
      xnn_external_value{2, out.data()},
  };

  ASSERT_EQ(xnn_reshape_runtime(runtime), xnn_status_success);
  ASSERT_EQ(xnn_setup_runtime_v2(runtime, 3, external_values.data()),
            xnn_status_success);

  ASSERT_EQ(xnn_invoke_runtime(runtime), xnn_status_success);

  if (ParallelLoopRunner* runner = GetParallelLoopRunner(threadpool)) {
    tsl::BlockUntilReady(runner->done_event());
    ASSERT_TRUE(runner->done_event().IsConcrete());
  }

  ASSERT_TRUE(absl::c_all_of(out, [&](float v) { return v == k; }));

  ASSERT_EQ(xnn_delete_runtime(runtime), xnn_status_success);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
  ASSERT_EQ(xnn_release_workspace(workspace), xnn_status_success);

  DestroyThreadPool(threadpool);
}

INSTANTIATE_TEST_SUITE_P(XnnThreadPool, XnnThreadPoolTest, testing::Bool(),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace xla::cpu
