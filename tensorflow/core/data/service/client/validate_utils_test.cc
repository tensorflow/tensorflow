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
#include "tensorflow/core/data/service/client/validate_utils.h"

#include <memory>

#include <gmock/gmock.h>
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

DataServiceParams GetDefaultParams() {
  DataServiceParams params;
  params.dataset_id = "dataset_id";
  params.processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  params.address = "localhost";
  params.protocol = "grpc";
  params.data_transfer_protocol = "grpc";
  params.metadata.set_cardinality(kUnknownCardinality);
  return params;
}

std::shared_ptr<DataServiceWorkerImpl> GetLocalWorker() {
  experimental::WorkerConfig config;
  config.set_protocol("grpc");
  config.set_dispatcher_address("localhost");
  config.set_worker_address("localhost");
  return std::make_shared<DataServiceWorkerImpl>(config);
}

TEST(ValidateUtilsTest, DefaultParams) {
  TF_EXPECT_OK(ValidateDataServiceParams(GetDefaultParams()));
}

TEST(ValidateUtilsTest, LocalWorkerSuccess) {
  DataServiceParams params = GetDefaultParams();
  LocalWorkers::Add("localhost", GetLocalWorker());
  params.target_workers = TARGET_WORKERS_LOCAL;
  TF_EXPECT_OK(ValidateDataServiceParams(params));
  LocalWorkers::Remove("localhost");
}

TEST(ValidateUtilsTest, NoLocalWorker) {
  DataServiceParams params = GetDefaultParams();
  params.target_workers = TARGET_WORKERS_LOCAL;
  EXPECT_THAT(
      ValidateDataServiceParams(params),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Local reads require local tf.data workers, but no local worker "
              "is found.")));
}

TEST(ValidateUtilsTest, NoLocalWorkerStaticSharding) {
  DataServiceParams params = GetDefaultParams();
  params.processing_mode.set_sharding_policy(ProcessingModeDef::FILE_OR_DATA);
  params.target_workers = TARGET_WORKERS_LOCAL;
  EXPECT_THAT(
      ValidateDataServiceParams(params),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Static sharding policy <FILE_OR_DATA> requires local tf.data "
              "workers, but no local worker is found.")));
}

TEST(ValidateUtilsTest, LocalReadDisallowsCoordinatedRead) {
  DataServiceParams params = GetDefaultParams();
  LocalWorkers::Add("localhost", GetLocalWorker());
  params.num_consumers = 1;
  params.consumer_index = 0;
  params.target_workers = TARGET_WORKERS_LOCAL;
  EXPECT_THAT(ValidateDataServiceParams(params),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  HasSubstr("Coordinated reads require non-local workers, but "
                            "`target_workers` is \"LOCAL\".")));
  LocalWorkers::Remove("localhost");
}

TEST(ValidateUtilsTest, CrossTrainerCacheSuccess) {
  DataServiceParams params = GetDefaultParams();
  params.job_name = "job_name";
  params.repetition = 1;
  params.metadata.set_cardinality(kInfiniteCardinality);
  params.cross_trainer_cache_options.emplace();
  params.cross_trainer_cache_options->set_trainer_id("trainer ID");
  TF_EXPECT_OK(ValidateDataServiceParams(params));
}

TEST(ValidateUtilsTest, CrossTrainerCacheRequiresJobName) {
  DataServiceParams params = GetDefaultParams();
  params.repetition = 1;
  params.metadata.set_cardinality(kInfiniteCardinality);
  params.cross_trainer_cache_options.emplace();
  params.cross_trainer_cache_options->set_trainer_id("trainer ID");
  EXPECT_THAT(
      ValidateDataServiceParams(params),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          "Cross-trainer caching requires named jobs. Got empty `job_name`."));
}

TEST(ValidateUtilsTest, CrossTrainerCacheRequiresInfiniteDataset) {
  DataServiceParams params = GetDefaultParams();
  params.job_name = "job_name";
  params.repetition = 1;
  params.metadata.set_cardinality(10);
  params.cross_trainer_cache_options.emplace();
  params.cross_trainer_cache_options->set_trainer_id("trainer ID");
  EXPECT_THAT(ValidateDataServiceParams(params),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  HasSubstr("Cross-trainer caching requires the input "
                            "dataset to be infinite.")));
}

TEST(ValidateUtilsTest, CrossTrainerCacheDisallowsRepetition) {
  DataServiceParams params = GetDefaultParams();
  params.job_name = "job_name";
  params.repetition = 5;
  params.metadata.set_cardinality(kInfiniteCardinality);
  params.cross_trainer_cache_options.emplace();
  params.cross_trainer_cache_options->set_trainer_id("trainer ID");
  EXPECT_THAT(
      ValidateDataServiceParams(params),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Cross-trainer caching requires infinite datasets and disallows "
              "multiple repetitions of the same dataset.")));
}

TEST(ValidateUtilsTest, CrossTrainerCacheDisallowsCoordinatedRead) {
  DataServiceParams params = GetDefaultParams();
  params.job_name = "job_name";
  params.repetition = 1;
  params.num_consumers = 1;
  params.consumer_index = 0;
  params.metadata.set_cardinality(kInfiniteCardinality);
  params.cross_trainer_cache_options.emplace();
  params.cross_trainer_cache_options->set_trainer_id("trainer ID");
  EXPECT_THAT(
      ValidateDataServiceParams(params),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Cross-trainer caching does not support coordinated reads.")));
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
