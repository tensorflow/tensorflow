/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/data_service.h"

#include <vector>

#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kProtocol[] = "grpc+local";

TEST(DataService, ParseParallelEpochsProcessingMode) {
  ProcessingMode mode;
  TF_ASSERT_OK(ParseProcessingMode("parallel_epochs", mode));
  EXPECT_EQ(mode, ProcessingMode::PARALLEL_EPOCHS);
}

TEST(DataService, ParseDistributedEpochProcessingMode) {
  ProcessingMode mode;
  TF_ASSERT_OK(ParseProcessingMode("distributed_epoch", mode));
  EXPECT_EQ(mode, ProcessingMode::DISTRIBUTED_EPOCH);
}

TEST(DataService, ParseInvalidProcessingMode) {
  ProcessingMode mode;
  Status s = ParseProcessingMode("invalid", mode);
  EXPECT_EQ(s.code(), error::Code::INVALID_ARGUMENT);
}

TEST(DataService, ProcessingModeToString) {
  EXPECT_EQ("parallel_epochs",
            ProcessingModeToString(ProcessingMode::PARALLEL_EPOCHS));
  EXPECT_EQ("distributed_epoch",
            ProcessingModeToString(ProcessingMode::DISTRIBUTED_EPOCH));
}

TEST(DataService, ParseTargetWorkers) {
  TF_ASSERT_OK_AND_ASSIGN(TargetWorkers target_workers,
                          ParseTargetWorkers("AUTO"));
  EXPECT_EQ(target_workers, TargetWorkers::AUTO);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers("Auto"));
  EXPECT_EQ(target_workers, TargetWorkers::AUTO);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers("ANY"));
  EXPECT_EQ(target_workers, TargetWorkers::ANY);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers("any"));
  EXPECT_EQ(target_workers, TargetWorkers::ANY);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers("LOCAL"));
  EXPECT_EQ(target_workers, TargetWorkers::LOCAL);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers("local"));
  EXPECT_EQ(target_workers, TargetWorkers::LOCAL);
  TF_ASSERT_OK_AND_ASSIGN(target_workers, ParseTargetWorkers(""));
  EXPECT_EQ(target_workers, TargetWorkers::AUTO);
}

TEST(DataService, ParseInvalidTargetWorkers) {
  EXPECT_TRUE(errors::IsInvalidArgument(ParseTargetWorkers("UNSET").status()));
}

TEST(DataService, TargetWorkersToString) {
  EXPECT_EQ(TargetWorkersToString(TargetWorkers::AUTO), "AUTO");
  EXPECT_EQ(TargetWorkersToString(TargetWorkers::ANY), "ANY");
  EXPECT_EQ(TargetWorkersToString(TargetWorkers::LOCAL), "LOCAL");
}

TEST(DataService, GetWorkers) {
  TestCluster cluster(1);
  TF_ASSERT_OK(cluster.Initialize());
  DataServiceDispatcherClient dispatcher(cluster.DispatcherAddress(),
                                         kProtocol);
  std::vector<WorkerInfo> workers;
  TF_EXPECT_OK(dispatcher.GetWorkers(workers));
  EXPECT_EQ(1, workers.size());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
