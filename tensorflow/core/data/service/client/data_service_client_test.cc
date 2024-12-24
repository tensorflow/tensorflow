/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/client/data_service_client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::RangeDataset;
using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAreArray;

DataServiceParams GetDataServiceParams(
    const std::string& dataset_id, const std::string& data_service_address,
    const ProcessingModeDef::ShardingPolicy sharding_policy) {
  DataServiceParams params;
  params.dataset_id = dataset_id;
  params.processing_mode.set_sharding_policy(sharding_policy);
  params.address = data_service_address;
  params.protocol = "grpc";
  params.data_transfer_protocol = "grpc";
  params.job_name = "test_job";
  params.repetition = 0;
  params.max_outstanding_requests = 100;
  params.task_refresh_interval = absl::Milliseconds(100);
  return params;
}

std::vector<int64_t> Range(const int64_t range) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

class TestDataServiceContext : public DataServiceContext {
 public:
  TestDataServiceContext() = default;
  ~TestDataServiceContext() override = default;

  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) override {
    return absl::WrapUnique(
        Env::Default()->StartThread({}, name, std::move(fn)));
  }

  MOCK_METHOD(void, RecordBufferEnqueue, (const std::vector<Tensor>& element),
              (override));
  MOCK_METHOD(void, RecordBufferDequeue, (const std::vector<Tensor>& element),
              (override));

  double GetTargetProcessingTimeNsec() const override { return 1.0e6; }
  int64_t UpdateMaxOutstandingRequests(int64_t max_outstanding_requests,
                                       int64_t new_size) override {
    return new_size;
  }
};

std::unique_ptr<TestDataServiceContext> GetTestDataServiceContext() {
  return std::make_unique<TestDataServiceContext>();
}

template <class T>
StatusOr<std::vector<T>> GetResults(DataServiceClient& client) {
  std::vector<T> results;
  while (true) {
    TF_ASSIGN_OR_RETURN(GetNextResult next,
                        client.GetNext(GetTestDataServiceContext));
    if (next.end_of_sequence) {
      return results;
    }
    results.push_back(next.tensors[0].unaligned_flat<T>().data()[0]);
  }
  return results;
}

template <class T>
StatusOr<T> GetNext(DataServiceClient& client) {
  TF_ASSIGN_OR_RETURN(GetNextResult next,
                      client.GetNext(GetTestDataServiceContext));
  if (next.end_of_sequence) {
    return errors::OutOfRange(
        "The tf.data service has reached the end of sequence");
  }
  return next.tensors[0].unaligned_flat<T>().data()[0];
}

TEST(DataServiceClientTest, NoSharding) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> test_dataset(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          test_dataset.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::OFF);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(/*accelerator_device_info=*/nullptr,
                                 /*allocator=*/nullptr));
  EXPECT_THAT(GetResults<int64_t>(client),
              IsOkAndHolds(ElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, DynamicSharding) {
  TestCluster test_cluster(/*num_workers=*/3);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> test_dataset(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          test_dataset.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::DYNAMIC);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(/*accelerator_device_info=*/nullptr,
                                 /*allocator=*/nullptr));
  EXPECT_THAT(GetResults<int64_t>(client),
              IsOkAndHolds(UnorderedElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, StaticSharding) {
  TestCluster test_cluster(/*num_workers=*/3);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> dataset_client(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          dataset_client.RegisterDataset(RangeDataset(10)));

  DataServiceParams params =
      GetDataServiceParams(dataset_id, test_cluster.DispatcherAddress(),
                           ProcessingModeDef::FILE_OR_DATA);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(/*accelerator_device_info=*/nullptr,
                                 /*allocator=*/nullptr));
  EXPECT_THAT(GetResults<int64_t>(client),
              IsOkAndHolds(UnorderedElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, RecordBufferEvents) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> test_dataset(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          test_dataset.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::OFF);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(/*accelerator_device_info=*/nullptr,
                                 /*allocator=*/nullptr));

  auto mock_context = std::make_unique<TestDataServiceContext>();
  TestDataServiceContext* ctx = mock_context.get();
  EXPECT_CALL(*ctx, RecordBufferEnqueue(_)).Times(AtLeast(1));
  EXPECT_CALL(*ctx, RecordBufferDequeue(_)).Times(AtLeast(1));

  TF_ASSERT_OK_AND_ASSIGN(GetNextResult next, client.GetNext([&mock_context]() {
    return std::move(mock_context);
  }));
  client.Cancel();
}

TEST(DataServiceClientTest, Cancel) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> dataset_client(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          dataset_client.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::OFF);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(/*accelerator_device_info=*/nullptr,
                                 /*allocator=*/nullptr));
  client.Cancel();
  EXPECT_THAT(client.GetNext(GetTestDataServiceContext),
              StatusIs(error::CANCELLED));
}

TEST(DataServiceClientTest, ValidationError) {
  DataServiceParams params = GetDataServiceParams(
      "dataset_id", "tf_data_service_address", ProcessingModeDef::OFF);
  params.target_workers = TARGET_WORKERS_LOCAL;
  DataServiceClient client(params);
  EXPECT_THAT(
      client.Initialize(/*accelerator_device_info=*/nullptr,
                        /*allocator=*/nullptr),
      StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Local reads require local tf.data workers, but no local worker "
              "is found.")));
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
