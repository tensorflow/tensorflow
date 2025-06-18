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
#include "tensorflow/core/common_runtime/all_to_all.h"

#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class AllToAllTest : public ::testing::Test {
 protected:
  std::unique_ptr<CollectiveTestEnv> test_env_;
};

TEST_F(AllToAllTest, Success) {
  test_env_ = CreateCollectiveTestEnv(/*num_workers*/ 1,
                                      /*num_devices_per_worker*/ 3, DEVICE_CPU);
  std::vector<Tensor> tensors = {
      test::AsTensor<double>({1., 2., 3.}),
      test::AsTensor<double>({4., 5., 6.}),
      test::AsTensor<double>({7., 8., 9.}),
  };
  BlockingCounter counter(3);
  for (int i = 0; i < 3; ++i) {
    SchedClosure([this, &tensors, i, &counter]() {
      auto col_params = CreateCollectiveParams(*test_env_, i, "AllToAll",
                                               ALL_TO_ALL_COLLECTIVE, DT_DOUBLE,
                                               tensors[i].shape());
      Device* device = nullptr;
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(
          col_params->group.members[i].device.name(), &device));
      TF_CHECK_OK(RunCollective(test_env_.get(), col_params.get(), device,
                                &tensors[i], &tensors[i]));
      counter.DecrementCount();
    });
  }
  counter.Wait();
  test::ExpectTensorEqual<double>(tensors[0],
                                  test::AsTensor<double>({1., 4., 7.}));
  test::ExpectTensorEqual<double>(tensors[1],
                                  test::AsTensor<double>({2., 5., 8.}));
  test::ExpectTensorEqual<double>(tensors[2],
                                  test::AsTensor<double>({3., 6., 9.}));
}

TEST_F(AllToAllTest, SuccessDifferentRank) {
  test_env_ = CreateCollectiveTestEnv(/*num_workers*/ 1,
                                      /*num_devices_per_worker*/ 3, DEVICE_CPU);
  std::vector<Tensor> tensors = {
      test::AsTensor<double>({1., 2., 3.}),
      test::AsTensor<double>({4., 5., 6.}),
      test::AsTensor<double>({7., 8., 9.}),
  };
  std::vector<std::vector<int32>> device_ranks = {{2, 1, 0}};
  BlockingCounter counter(3);
  for (int i = 0; i < 3; ++i) {
    SchedClosure([this, &tensors, &device_ranks, i, &counter]() {
      auto col_params = CreateCollectiveParams(
          *test_env_, i, "AllToAll", ALL_TO_ALL_COLLECTIVE, DT_DOUBLE,
          tensors[i].shape(), device_ranks);
      Device* device = nullptr;
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(
          col_params->group.members[i].device.name(), &device));
      TF_CHECK_OK(RunCollective(test_env_.get(), col_params.get(), device,
                                &tensors[i], &tensors[i]));
      counter.DecrementCount();
    });
  }
  counter.Wait();
  test::ExpectTensorEqual<double>(tensors[0],
                                  test::AsTensor<double>({7., 4., 1.}));
  test::ExpectTensorEqual<double>(tensors[1],
                                  test::AsTensor<double>({8., 5., 2.}));
  test::ExpectTensorEqual<double>(tensors[2],
                                  test::AsTensor<double>({9., 6., 3.}));
}

TEST_F(AllToAllTest, Failure) {
  test_env_ = CreateCollectiveTestEnv(/*num_workers*/ 1,
                                      /*num_devices_per_worker*/ 3, DEVICE_CPU);
  test_env_->remote_access->set_fail_after(1);
  std::vector<Tensor> tensors = {
      test::AsTensor<double>({1., 2., 3.}),
      test::AsTensor<double>({4., 5., 6.}),
      test::AsTensor<double>({7., 8., 9.}),
  };
  int num_failures = 0;
  mutex mu;
  BlockingCounter counter(3);
  for (int i = 0; i < 3; ++i) {
    SchedClosure([this, &mu, &num_failures, &tensors, i, &counter]() {
      auto col_params = CreateCollectiveParams(*test_env_, i, "AllToAll",
                                               ALL_TO_ALL_COLLECTIVE, DT_DOUBLE,
                                               tensors[i].shape());
      Device* device = nullptr;
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(
          col_params->group.members[i].device.name(), &device));
      absl::Status status = RunCollective(test_env_.get(), col_params.get(),
                                          device, &tensors[i], &tensors[i]);
      if (!status.ok()) {
        mutex_lock l(mu);
        ++num_failures;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();
  // Failures are not guaranteed to propagate since the communication is P2P.
  // One worker can succeed while part of the communication fail.
  EXPECT_GT(num_failures, 0);
}

TEST_F(AllToAllTest, WrongFirstDimensionSize) {
  test_env_ = CreateCollectiveTestEnv(/*num_workers*/ 1,
                                      /*num_devices_per_worker*/ 3, DEVICE_CPU);
  std::vector<Tensor> tensors = {
      test::AsTensor<double>({1., 2.}),
      test::AsTensor<double>({4., 5.}),
      test::AsTensor<double>({7., 8.}),
  };
  BlockingCounter counter(3);
  for (int i = 0; i < 3; ++i) {
    SchedClosure([this, &tensors, i, &counter]() {
      auto col_params = CreateCollectiveParams(*test_env_, i, "AllToAll",
                                               ALL_TO_ALL_COLLECTIVE, DT_DOUBLE,
                                               tensors[i].shape());
      Device* device = nullptr;
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(
          col_params->group.members[i].device.name(), &device));
      absl::Status status = RunCollective(test_env_.get(), col_params.get(),
                                          device, &tensors[i], &tensors[i]);
      counter.DecrementCount();
      EXPECT_TRUE(absl::IsInvalidArgument(status));
    });
  }
  counter.Wait();
}

}  // namespace
}  // namespace tensorflow
