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

#include "tensorflow/core/kernels/batch_kernels.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/batch_kernel_test_util.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace {

using PerModelData = serving::WarmupStateRegistry::PerModelData;

class BatchFunctionKernelTest : public test_util::BatchFunctionKernelTestBase {
};

TEST_P(BatchFunctionKernelTest, EnableAdaptiveScheduler) {
  const bool adaptive_scheduler_enabled = GetParam();

  TF_EXPECT_OK(Init(adaptive_scheduler_enabled));

  BatchFunctionKernel *batch_kernel =
      dynamic_cast<BatchFunctionKernel *>(op_kernel());
  EXPECT_EQ(adaptive_scheduler_enabled,
            test_util::BatchFunctionKernelTestAccess(batch_kernel)
                .enable_adaptive_batch_threads());
}

INSTANTIATE_TEST_SUITE_P(Params, BatchFunctionKernelTest, ::testing::Bool());

class BatchFunctionKernelParallelWarmupTestState : public OpsTestBase {
 public:
  // Init test fixture with a batch kernel instance.
  Status Init(bool enable_splitting) {
    static auto *const cpu_device = []() {
      auto device =
          DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
      return device.release();
    }();

    // Override the per-test/per-op device with a global device so that it can
    // be shared between ops.
    device_ = cpu_device;

    NameAttrList f;
    f.set_name("BatchFunctionKernelParallelWarmupTestStateFunc");
    FunctionDef func = FunctionDefHelper::Create(
        // function_name
        f.name(),
        // in_def
        {"x:int64"},
        // out_def
        {"o:int64"},
        // attr_def
        {},
        // node_def
        {{{"o"},
          "EnsureShape",
          {"x"},
          {{"T", DataType::DT_INT64}, {"shape", TensorShape({2})}}}},
        // ret_def
        {{"o", "o:output"}});
    TF_RETURN_IF_ERROR(flib_def_->AddFunctionDef(func));

    pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions(),
        /*thread_pool=*/nullptr, /*parent=*/nullptr,
        /*session_metadata=*/nullptr,
        Rendezvous::Factory{[](const int64_t, const DeviceMgr *device_mgr,
                               tsl::core::RefCountPtr<Rendezvous> *r) {
          *r = tsl::core::RefCountPtr<Rendezvous>(
              new IntraProcessRendezvous(device_mgr));
          return absl::OkStatus();
        }});

    std::vector<NodeDefBuilder::NodeOut> inputs(
        {NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64})});
    TF_CHECK_OK(NodeDefBuilder("BatchTPUInput", "BatchFunction")
                    .Attr("max_batch_size", enable_splitting ? 16 : 8)
                    .Attr("num_batch_threads", 8)
                    .Attr("allowed_batch_sizes", {2, 4, 8})
                    .Attr("batch_timeout_micros", 1000000)
                    .Attr("max_enqueued_batches", 10)
                    .Attr("enable_large_batch_splitting", true)
                    .Attr("low_priority_max_batch_size", 64)
                    .Attr("low_priority_batch_timeout_micros", 8000)
                    .Attr("low_priority_allowed_batch_sizes", {32, 64})
                    .Attr("low_priority_max_enqueued_batches", 1000)
                    .Attr("Tin", {DataType::DT_INT64})
                    .Input(inputs)
                    .Attr("Tcaptured", std::vector<DataType>{})
                    .Input(std::vector<NodeDefBuilder::NodeOut>{})
                    .Attr("Tout", std::vector<DataType>{DT_INT64})
                    .Attr("f", f)
                    .Finalize(node_def()));
    return InitOp();
  }

  void TestBody() override {}
};

class BatchFunctionKernelParallelWarmupTest
    : public ::testing::TestWithParam<bool> {};

TEST_P(BatchFunctionKernelParallelWarmupTest, ParallelWarmup) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);
  serving::WarmupStateRegistry::Key key(session_metadata.name(),
                                        session_metadata.version());

  int num_requests = 16;

  bool enable_splitting = GetParam();

  {
    // Setting the state to warmup disables batching in the BatchFunction op. We
    // are checking this behavior by checking the tensor shape inside batch
    // function is the same as the input tensor shape using EnsureShape op.
    auto per_model_data = std::make_unique<PerModelData>();
    auto handle = serving::GetGlobalWarmupStateRegistry().Register(
        key, std::move(per_model_data));
    tsl::BlockingCounter blocking_counter(num_requests);
    for (int i = 0; i < num_requests; ++i) {
      Env::Default()->SchedClosure([&]() {
        BatchFunctionKernelParallelWarmupTestState test;
        test.set_session_metadata(session_metadata);
        TF_CHECK_OK(test.Init(enable_splitting));
        test.AddInputFromList<int64_t>(TensorShape({2}), {123, 456});
        TF_CHECK_OK(test.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(*test.GetOutput(0),
                                         test::AsTensor<int64_t>({123, 456}));
        blocking_counter.DecrementCount();
      });
    }
    // Note this times out after 60s, so `batch_timeout_micros` and `batch_size`
    // need to be set accordingly.
    blocking_counter.Wait();
  }

  EXPECT_FALSE(serving::GetGlobalWarmupStateRegistry().Lookup(key));
  {
    tsl::BlockingCounter blocking_counter(num_requests);
    for (int i = 0; i < num_requests; ++i) {
      Env::Default()->SchedClosure([&]() {
        BatchFunctionKernelParallelWarmupTestState test;
        test.set_session_metadata(session_metadata);
        TF_CHECK_OK(test.Init(enable_splitting));
        test.AddInputFromList<int64_t>(TensorShape({2}), {123, 456});
        // We expect requests to be batched together when the warm-up mode is
        // turned off, which will make the execution fail at `EnsureShape`.
        EXPECT_FALSE(test.RunOpKernel().ok());

        blocking_counter.DecrementCount();
      });
    }
    blocking_counter.Wait();
  }
}

INSTANTIATE_TEST_SUITE_P(BatchFunctionKernelParallelWarmupTestSuite,
                         BatchFunctionKernelParallelWarmupTest,
                         ::testing::Bool());
}  // namespace
}  // namespace tensorflow
