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

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
constexpr const char kProtocol[] = "grpc+local";
}

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

TEST(DataService, GetWorkers) {
  TestCluster cluster(1);
  TF_ASSERT_OK(cluster.Initialize());
  DataServiceDispatcherClient dispatcher(cluster.DispatcherAddress(),
                                         kProtocol);
  std::vector<WorkerInfo> workers;
  TF_EXPECT_OK(dispatcher.GetWorkers(workers));
  EXPECT_EQ(1, workers.size());
}

}  // namespace data
}  // namespace tensorflow
