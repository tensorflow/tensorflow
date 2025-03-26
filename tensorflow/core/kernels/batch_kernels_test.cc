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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/criticality.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/batch_kernel_test_util.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/refcount.h"

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

class SharedBatchFunctionTestState : public OpsTestBase {
 public:
  // Init test fixture with a batch kernel instance.
  void CreateFunctionLibraryRuntime() {
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
  }

 protected:
  // Create common batch function op for testing.
  absl::StatusOr<NodeDefBuilder> CreateBatchFunctionBuilder(
      const std::vector<int> &allowed_batch_sizes, int max_batch_size,
      absl::string_view padding_policy,
      const TensorShape &expected_output_shape) {
    NameAttrList f;
    f.set_name("ShapeEnforcingFunction");
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
          {{"T", DataType::DT_INT64}, {"shape", expected_output_shape}}}},
        // ret_def
        {{"o", "o:output"}});
    TF_RETURN_IF_ERROR(flib_def_->AddFunctionDef(func));
    SharedBatchFunctionTestState::CreateFunctionLibraryRuntime();

    std::vector<NodeDefBuilder::NodeOut> inputs(
        {NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64})});
    return NodeDefBuilder(absl::StrCat("BatchTPUInput", padding_policy),
                          "BatchFunction")
        .Attr("max_batch_size", max_batch_size)
        .Attr("num_batch_threads", 8)
        .Attr("allowed_batch_sizes", allowed_batch_sizes)
        .Attr("batch_timeout_micros", 1000000)
        .Attr("max_enqueued_batches", 10)
        .Attr("enable_large_batch_splitting", true)
        .Attr("batch_padding_policy", padding_policy)
        .Attr("Tin", {DataType::DT_INT64})
        .Input(inputs)
        .Attr("Tcaptured", std::vector<DataType>{})
        .Input(std::vector<NodeDefBuilder::NodeOut>{})
        .Attr("Tout", std::vector<DataType>{DT_INT64})
        .Attr("f", f);
  }
};

class BatchFunctionTestState : public SharedBatchFunctionTestState {
 public:
  // Init test fixture with a batch kernel instance. The caller guarantees that
  // the device pointer is valid throughout the life of this class.
  absl::Status Init(Device *device, bool enable_low_priority_queue,
                    absl::string_view mixed_priority_policy,
                    int64_t expected_batch_size) {
    // Override the per-test/per-op device with a given device so that it can
    // be shared between ops.
    device_ = device;

    const TensorShape expected_output_shape({expected_batch_size, 2});
    TF_ASSIGN_OR_RETURN(
        NodeDefBuilder builder,
        CreateBatchFunctionBuilder({4, 8}, 8, "PAD_UP", expected_output_shape));
    TF_RETURN_IF_ERROR(builder
                           .Attr("low_priority_max_batch_size",
                                 enable_low_priority_queue ? 8 : 0)
                           .Attr("low_priority_batch_timeout_micros",
                                 enable_low_priority_queue ? 2000000 : 0)
                           .Attr("low_priority_allowed_batch_sizes",
                                 enable_low_priority_queue
                                     ? std::vector<int>{4, 8}
                                     : std::vector<int>())
                           .Attr("low_priority_max_enqueued_batches",
                                 enable_low_priority_queue ? 2 : 0)
                           .Attr("mixed_priority_policy", mixed_priority_policy)
                           .Finalize(node_def()));

    return OpsTestBase::InitOp();
  }

  void TestBody() override {}
};

class BatchFunctionTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    // The device needs to be shared in each test case and within each test case
    // only.
    cpu_device_ =
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
  }
  std::unique_ptr<Device> cpu_device_;
};

TEST_P(BatchFunctionTest, BatchingWorksWithoutCriticality) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  bool enable_low_priority_queue = GetParam();
  {
    tsl::BlockingCounter blocking_counter(8);
    // 8 threads run the batch op with no explicit criticality set. They are
    // eventually batched to form a tensor with [8, 2] shape which is verified
    // within the function.
    for (int i = 0; i < 8; ++i) {
      Env::Default()->SchedClosure([&]() {
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kCritical);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/8));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

TEST_P(BatchFunctionTest, PaddingWorksWithoutCriticality) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  bool enable_low_priority_queue = GetParam();
  {
    tsl::BlockingCounter blocking_counter(2);
    // 2 threads run the batch op with no explicit criticality set. They are
    // eventually batched and padded to form a tensor with [4, 2] shape which is
    // verified within the function.
    for (int i = 0; i < 2; ++i) {
      Env::Default()->SchedClosure([&]() {
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kCritical);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

#if defined(PLATFORM_GOOGLE)
TEST_P(BatchFunctionTest,
       LowPriorityTaskPaddingHighPriorityBatchUptoMaxBatchSize) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  bool enable_low_priority_queue = GetParam();
  {
    tsl::BlockingCounter blocking_counter(8);
    // 4 threads run the batch op with critical plus and 4 threads run the batch
    // op with sheddable. They are eventually batched to form a tensor with [8,
    // 2] shape which is verified within the function.
    for (int i = 0; i < 4; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kCriticalPlus);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kCriticalPlus);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/8));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    for (int i = 0; i < 4; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kSheddable);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/8));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

TEST_P(BatchFunctionTest,
       LowPriorityTaskPaddingHighPriorityBatchWithExtraPadding) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  bool enable_low_priority_queue = GetParam();
  {
    tsl::BlockingCounter blocking_counter(2);
    // 1 thread run the batch op with critical plus and 1 threads run the batch
    // op with sheddable. They are eventually batched and padded to form a
    // tensor with [4, 2] shape which is verified within the function.
    Env::Default()->SchedClosure([&]() {
      tsl::criticality::ScopedCriticality scoped_criticality(
          tsl::criticality::Criticality::kCriticalPlus);
      ASSERT_EQ(tsl::criticality::GetCriticality(),
                tsl::criticality::Criticality::kCriticalPlus);

      BatchFunctionTestState test_state;
      test_state.set_session_metadata(session_metadata);
      TF_ASSERT_OK(
          test_state.Init(cpu_device_.get(), enable_low_priority_queue,
                          serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
                          /*expected_batch_size=*/4));
      test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
      TF_EXPECT_OK(test_state.RunOpKernel());

      test::ExpectTensorEqual<int64_t>(
          *test_state.GetOutput(0),
          test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
      blocking_counter.DecrementCount();
    });

    Env::Default()->SchedClosure([&]() {
      tsl::criticality::ScopedCriticality scoped_criticality(
          tsl::criticality::Criticality::kSheddable);
      ASSERT_EQ(tsl::criticality::GetCriticality(),
                tsl::criticality::Criticality::kSheddable);

      BatchFunctionTestState test_state;
      test_state.set_session_metadata(session_metadata);
      TF_ASSERT_OK(
          test_state.Init(cpu_device_.get(), enable_low_priority_queue,
                          serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
                          /*expected_batch_size=*/4));
      test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
      TF_EXPECT_OK(test_state.RunOpKernel());

      test::ExpectTensorEqual<int64_t>(
          *test_state.GetOutput(0),
          test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
      blocking_counter.DecrementCount();
    });

    blocking_counter.Wait();
  }
}

TEST_P(BatchFunctionTest,
       LowPriorityTaskPaddingHighPriorityBatchUptoNextAllowedBatchSize) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  bool enable_low_priority_queue = GetParam();
  {
    tsl::BlockingCounter blocking_counter(4);
    // 2 threads run the batch op with critical plus and 2 threads run the batch
    // op with sheddable. They are eventually batched to form a tensor with [4,
    // 2] shape which is verified within the function.
    for (int i = 0; i < 2; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kCriticalPlus);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kCriticalPlus);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue,
            /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    for (int i = 0; i < 2; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kSheddable);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(), enable_low_priority_queue,
            serving::kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue,
            /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}
#endif

INSTANTIATE_TEST_SUITE_P(BatchFunctionTest, BatchFunctionTest,
                         ::testing::Bool());

#if defined(PLATFORM_GOOGLE)
TEST_F(BatchFunctionTest, HighPriorityBatchNotPaddedWithLowPriorityTasks) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  {
    tsl::BlockingCounter blocking_counter(8);
    // 4 threads run the batch op with critical plus and 4 threads run the batch
    // op with sheddable. They each get batched separately to form tensors with
    // [4, 2] shape which is verified within the function.
    for (int i = 0; i < 4; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kCriticalPlus);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kCriticalPlus);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(cpu_device_.get(),
                                     /*enable_low_priority_queue=*/true,
                                     serving::kPriorityIsolationAttrValue,
                                     /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    for (int i = 0; i < 4; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kSheddable);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(cpu_device_.get(),
                                     /*enable_low_priority_queue=*/true,
                                     serving::kPriorityIsolationAttrValue,
                                     /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

TEST_F(BatchFunctionTest, LowPriorityOnlyBatchAtMaxLowPriorityBatchSize) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  {
    tsl::BlockingCounter blocking_counter(8);
    // 8 threads run the batch op with sheddable. They are eventually batched to
    // form a tensor with [8, 2] shape, which is verified within the function,
    // since the low priority max batch size is set to 8.
    for (int i = 0; i < 8; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kSheddable);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(),
            /*enable_low_priority_queue=*/true,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/8));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

TEST_F(BatchFunctionTest, LowPriorityBatchPaddedToLowPriorityAllowedBatchSize) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  {
    tsl::BlockingCounter blocking_counter(2);
    // 2 threads run the batch op with sheddable. They are eventually batched
    // and padded to form a tensor with [4, 2] shape, which is verified within
    // the function, since the low priority allowed batch size is set to [4, 8].
    for (int i = 0; i < 2; ++i) {
      Env::Default()->SchedClosure([&]() {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        ASSERT_EQ(tsl::criticality::GetCriticality(),
                  tsl::criticality::Criticality::kSheddable);

        BatchFunctionTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_ASSERT_OK(test_state.Init(
            cpu_device_.get(),
            /*enable_low_priority_queue=*/true,
            serving::kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*expected_batch_size=*/4));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {234, 567});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({234, 567}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}
#endif

class BatchFunctionKernelParallelWarmupTestState
    : public SharedBatchFunctionTestState {
 public:
  // Init test fixture with a batch kernel instance.
  absl::Status Init(bool enable_splitting) {
    static auto *const cpu_device = []() {
      auto device =
          DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
      return device.release();
    }();

    // Override the per-test/per-op device with a global device so that it can
    // be shared between ops.
    device_ = cpu_device;

    const TensorShape expected_output_shape({2});
    TF_ASSIGN_OR_RETURN(
        NodeDefBuilder builder,
        CreateBatchFunctionBuilder({2, 4, 8}, enable_splitting ? 16 : 8,
                                   "PAD_UP", expected_output_shape));
    TF_RETURN_IF_ERROR(builder.Finalize(node_def()));

    return OpsTestBase::InitOp();
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

class BatchFunctionKernelPaddingTestState
    : public SharedBatchFunctionTestState {
 public:
  // Init test fixture with a batch kernel instance.
  absl::Status Init(absl::string_view padding_policy, int expected_batch_size) {
    static auto *const cpu_device = []() {
      auto device =
          DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
      return device.release();
    }();

    // Override the per-test/per-op device with a global device so that it can
    // be shared between ops.
    device_ = cpu_device;

    const TensorShape expected_output_shape({expected_batch_size, 2});
    TF_RETURN_IF_ERROR(CreateBatchFunctionBuilder({4, 8}, 8, padding_policy,
                                                  expected_output_shape)
                           ->Finalize(node_def()));

    return OpsTestBase::InitOp();
  }

  void TestBody() override {}
};

class BatchFunctionKernelPaddingTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(BatchFunctionKernelPaddingTest, PadUp) {
  SessionMetadata session_metadata;
  session_metadata.set_name("test_model");
  session_metadata.set_version(123);

  // Send 5 requests in parallel and check that the given batch padding
  // policy behaves as expected.
  int64_t num_requests = 5;
  int64_t expected_batch_size = 0;
  std::string padding_policy = GetParam();
  if (padding_policy == "PAD_UP") {
    expected_batch_size = 8;
  } else if (padding_policy == "BATCH_DOWN") {
    expected_batch_size = 4;
  } else if (padding_policy == "MINIMIZE_TPU_COST_PER_REQUEST") {
    expected_batch_size = 8;
  } else {
    FAIL() << "Unsupported padding policy: " << padding_policy;
  }

  {
    tsl::BlockingCounter blocking_counter(num_requests);
    for (int i = 0; i < num_requests; ++i) {
      Env::Default()->SchedClosure([&]() {
        BatchFunctionKernelPaddingTestState test_state;
        test_state.set_session_metadata(session_metadata);
        TF_CHECK_OK(test_state.Init(padding_policy, expected_batch_size));
        test_state.AddInputFromList<int64_t>(TensorShape({1, 2}), {123, 456});
        TF_EXPECT_OK(test_state.RunOpKernel());

        test::ExpectTensorEqual<int64_t>(
            *test_state.GetOutput(0),
            test::AsTensor<int64_t>({123, 456}, TensorShape({1, 2})));
        blocking_counter.DecrementCount();
      });
    }

    blocking_counter.Wait();
  }
}

INSTANTIATE_TEST_SUITE_P(BatchFunctionKernelPaddingTestSuite,
                         BatchFunctionKernelPaddingTest,
                         ::testing::Values("PAD_UP", "BATCH_DOWN",
                                           "MINIMIZE_TPU_COST_PER_REQUEST"));

}  // namespace
}  // namespace tensorflow
