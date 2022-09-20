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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::RangeDataset;
using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
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
  params.task_refresh_interval_ms = 100;
  return params;
}

std::vector<int64_t> Range(const int64_t range) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < range; ++i) {
    result.push_back(i);
  }
  return result;
}

class TestContext {
 public:
  explicit TestContext() {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 1});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));

    FunctionDefLibrary proto;
    lib_def_ = std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
                                                           proto);

    OptimizerOptions opts;
    pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts);
    runner_ = [](const std::function<void()>& fn) { fn(); };
    params_.function_library = pflr_->GetFLR("/device:CPU:0");
    params_.device = device_mgr_->ListDevices()[0];
    params_.runner = &runner_;
    op_ctx_ = std::make_unique<OpKernelContext>(&params_, 0);
    iter_ctx_ = std::make_unique<IteratorContext>(op_ctx_.get());
  }

  IteratorContext* iterator_ctx() const { return iter_ctx_.get(); }

 private:
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::function<void(std::function<void()>)> runner_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> op_ctx_;
  std::unique_ptr<IteratorContext> iter_ctx_;
};

class TestDataServiceContext : public DataServiceContext {
 public:
  TestDataServiceContext() = default;
  ~TestDataServiceContext() override = default;

  void RecordBufferEnqueue(const std::vector<Tensor>& element) override {}
  void RecordBufferDequeue(const std::vector<Tensor>& element) override {}
};

std::unique_ptr<TestDataServiceContext> GetTestDataServiceContext() {
  return std::make_unique<TestDataServiceContext>();
}

template <class T>
StatusOr<std::vector<T>> GetResults(DataServiceClient& client,
                                    IteratorContext* ctx) {
  std::vector<T> results;
  while (true) {
    TF_ASSIGN_OR_RETURN(GetNextResult next,
                        client.GetNext(ctx, GetTestDataServiceContext));
    if (next.end_of_sequence) {
      return results;
    }
    results.push_back(next.tensors[0].unaligned_flat<T>().data()[0]);
  }
  return results;
}

template <class T>
StatusOr<T> GetNext(DataServiceClient& client, IteratorContext* ctx) {
  TF_ASSIGN_OR_RETURN(GetNextResult next,
                      client.GetNext(ctx, GetTestDataServiceContext));
  if (next.end_of_sequence) {
    return errors::OutOfRange(
        "The tf.data service has reached the end of sequence");
  }
  return next.tensors[0].unaligned_flat<T>().data()[0];
}

TEST(DataServiceClientTest, NoSharding) {
  TestContext ctx;
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> test_dataset(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          test_dataset.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::OFF);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(ctx.iterator_ctx()));
  EXPECT_THAT(GetResults<int64_t>(client, ctx.iterator_ctx()),
              IsOkAndHolds(ElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, DynamicSharding) {
  TestContext ctx;
  TestCluster test_cluster(/*num_workers=*/3);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> test_dataset(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          test_dataset.RegisterDataset(RangeDataset(10)));

  DataServiceParams params = GetDataServiceParams(
      dataset_id, test_cluster.DispatcherAddress(), ProcessingModeDef::DYNAMIC);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(ctx.iterator_ctx()));
  EXPECT_THAT(GetResults<int64_t>(client, ctx.iterator_ctx()),
              IsOkAndHolds(UnorderedElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, StaticSharding) {
  TestContext ctx;
  TestCluster test_cluster(/*num_workers=*/3);
  TF_ASSERT_OK(test_cluster.Initialize());
  DatasetClient<int64_t> dataset_client(test_cluster);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_id,
                          dataset_client.RegisterDataset(RangeDataset(10)));

  DataServiceParams params =
      GetDataServiceParams(dataset_id, test_cluster.DispatcherAddress(),
                           ProcessingModeDef::FILE_OR_DATA);
  DataServiceClient client(params);
  TF_ASSERT_OK(client.Initialize(ctx.iterator_ctx()));
  EXPECT_THAT(GetResults<int64_t>(client, ctx.iterator_ctx()),
              IsOkAndHolds(UnorderedElementsAreArray(Range(10))));
  client.Cancel();
}

TEST(DataServiceClientTest, ValidationError) {
  TestContext ctx;
  DataServiceParams params = GetDataServiceParams(
      "dataset_id", "tf_data_service_address", ProcessingModeDef::OFF);
  params.target_workers = TARGET_WORKERS_LOCAL;
  DataServiceClient client(params);
  EXPECT_THAT(
      client.Initialize(ctx.iterator_ctx()),
      StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Local reads require local tf.data workers, but no local worker "
              "is found.")));
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
