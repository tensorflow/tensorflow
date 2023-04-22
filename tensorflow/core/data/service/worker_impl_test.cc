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
#include "tensorflow/core/data/service/worker_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

class LocalWorkersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_cluster_ = absl::make_unique<TestCluster>(/*num_workers=*/0);
    TF_ASSERT_OK(test_cluster_->Initialize());
  }

  std::unique_ptr<TestCluster> test_cluster_;
};

TEST_F(LocalWorkersTest, AddRemoveLocalWorkers) {
  EXPECT_TRUE(LocalWorkers::Empty());
  TF_ASSERT_OK(test_cluster_->AddWorker());
  TF_ASSERT_OK(test_cluster_->AddWorker());
  TF_ASSERT_OK(test_cluster_->AddWorker());
  std::vector<std::string> worker_addresses = {test_cluster_->WorkerAddress(0),
                                               test_cluster_->WorkerAddress(1),
                                               test_cluster_->WorkerAddress(2)};

  EXPECT_FALSE(LocalWorkers::Empty());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[0]), NotNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[1]), NotNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[2]), NotNull());

  test_cluster_->StopWorker(0);
  EXPECT_FALSE(LocalWorkers::Empty());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[0]), IsNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[1]), NotNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[2]), NotNull());

  test_cluster_->StopWorkers();
  EXPECT_TRUE(LocalWorkers::Empty());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[0]), IsNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[1]), IsNull());
  EXPECT_THAT(LocalWorkers::Get(worker_addresses[2]), IsNull());
}

TEST_F(LocalWorkersTest, NoLocalWorker) {
  EXPECT_TRUE(LocalWorkers::Empty());
  EXPECT_THAT(LocalWorkers::Get(/*worker_address=*/""), IsNull());
  EXPECT_THAT(LocalWorkers::Get(/*worker_address=*/"Invalid address"),
              IsNull());
  EXPECT_TRUE(LocalWorkers::Empty());
  LocalWorkers::Remove(/*worker_address=*/"");
  LocalWorkers::Remove(/*worker_address=*/"Invalid address");
  EXPECT_TRUE(LocalWorkers::Empty());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
