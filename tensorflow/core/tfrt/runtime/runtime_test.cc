/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/runtime/runtime.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/common_runtime/cost_util.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::tensorflow::RequestCost;
using ::tensorflow::RequestCostAccessor;
using ::testing::NotNull;

class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override {
    static RequestCost* request_cost = new RequestCost();
    return request_cost;
  }
};

class RuntimeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // setenv needs to be called before the RequestCostAccessor is registered.
    setenv("TF_REQUEST_COST_ACCESSOR_TYPE", "test", 1 /*overwrite*/);
    tensorflow::RequestCostAccessorRegistry::RegisterRequestCostAccessor(
        "test", []() { return std::make_unique<TestRequestCostAccessor>(); });
  }
};

TEST_F(RuntimeTest, GlobalRuntimeWorks) {
  // Before SetGlobalRuntime, it is null.
  EXPECT_EQ(GetGlobalRuntime(), nullptr);
  // After SetGlobalRuntime, it is not null.
  SetGlobalRuntime(Runtime::Create(/*num_inter_op_threads=*/4));
  EXPECT_NE(GetGlobalRuntime(), nullptr);
  // It is only allocated once.
  EXPECT_EQ(GetGlobalRuntime(), GetGlobalRuntime());
}

TEST_F(RuntimeTest, CreateRequestQueue) {
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime =
      Runtime::Create(WrapDefaultWorkQueue(tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/1, /*num_blocking_queues=*/1)));
  absl::StatusOr<std::unique_ptr<WorkQueueInterface>> work_queue =
      runtime->CreateRequestQueue(/*request_id=*/100);

  const std::string kGetWorkQueueDelayUsecsMetric =
      "get_work_queue_delay_usecs";
  std::unique_ptr<tensorflow::RequestCostAccessor> cost_accessor =
      tensorflow::CreateRequestCostAccessor();
  ASSERT_THAT(cost_accessor, NotNull());
  ASSERT_THAT(cost_accessor->GetRequestCost(), NotNull());
  const absl::flat_hash_map<std::string, double>& metrics =
      cost_accessor->GetRequestCost()->GetMetrics();
  EXPECT_TRUE(metrics.contains(kGetWorkQueueDelayUsecsMetric));
  EXPECT_GT(metrics.at(kGetWorkQueueDelayUsecsMetric), 0);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
