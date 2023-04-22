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
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kProtocol[] = "grpc";

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
  EXPECT_THAT(ParseProcessingMode("invalid", mode),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST(DataService, ProcessingModeToString) {
  EXPECT_EQ("parallel_epochs",
            ProcessingModeToString(ProcessingMode::PARALLEL_EPOCHS));
  EXPECT_EQ("distributed_epoch",
            ProcessingModeToString(ProcessingMode::DISTRIBUTED_EPOCH));
}

TEST(DataService, ParseTargetWorkers) {
  EXPECT_THAT(ParseTargetWorkers("AUTO"),
              testing::IsOkAndHolds(TargetWorkers::AUTO));
  EXPECT_THAT(ParseTargetWorkers("Auto"),
              testing::IsOkAndHolds(TargetWorkers::AUTO));
  EXPECT_THAT(ParseTargetWorkers("ANY"),
              testing::IsOkAndHolds(TargetWorkers::ANY));
  EXPECT_THAT(ParseTargetWorkers("any"),
              testing::IsOkAndHolds(TargetWorkers::ANY));
  EXPECT_THAT(ParseTargetWorkers("LOCAL"),
              testing::IsOkAndHolds(TargetWorkers::LOCAL));
  EXPECT_THAT(ParseTargetWorkers("local"),
              testing::IsOkAndHolds(TargetWorkers::LOCAL));
  EXPECT_THAT(ParseTargetWorkers(""),
              testing::IsOkAndHolds(TargetWorkers::AUTO));
}

TEST(DataService, ParseInvalidTargetWorkers) {
  EXPECT_THAT(ParseTargetWorkers("UNSET"),
              testing::StatusIs(error::INVALID_ARGUMENT));
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
