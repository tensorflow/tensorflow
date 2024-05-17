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
#include "tensorflow/core/data/service/client/utils.h"

#include <optional>
#include <string>

#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::EqualsProto;
using ::tensorflow::data::testing::RangeDataset;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(UtilsTest, GetDataServiceMetadata) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  DataServiceDispatcherClient dispatcher_client(
      test_cluster.DispatcherAddress(), "grpc");
  DataServiceMetadata metadata;
  metadata.set_compression(DataServiceMetadata::COMPRESSION_SNAPPY);
  metadata.set_cardinality(100);

  std::string dataset_id;
  TF_ASSERT_OK(dispatcher_client.RegisterDataset(
      RangeDataset(10), metadata, /*requested_dataset_id=*/std::nullopt,
      dataset_id));
  EXPECT_THAT(GetDataServiceMetadata(dataset_id,
                                     test_cluster.DispatcherAddress(), "grpc"),
              IsOkAndHolds(EqualsProto(metadata)));
}

TEST(UtilsTest, GetDataServiceMetadataNotFound) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  EXPECT_THAT(GetDataServiceMetadata(/*dataset_id=*/"not found",
                                     test_cluster.DispatcherAddress(), "grpc"),
              StatusIs(error::NOT_FOUND));
}

TEST(UtilsTest, GetDataServiceConfig) {
  TestCluster test_cluster(/*num_workers=*/1);
  TF_ASSERT_OK(test_cluster.Initialize());
  TF_ASSERT_OK_AND_ASSIGN(
      DataServiceConfig service_config,
      GetDataServiceConfig(test_cluster.DispatcherAddress(), "grpc"));
  EXPECT_EQ(service_config.deployment_mode(), DEPLOYMENT_MODE_COLOCATED);
}

TEST(UtilsTest, GetValidatedCompression) {
  DataServiceMetadata metadata;
  metadata.set_compression(DataServiceMetadata::COMPRESSION_SNAPPY);
  EXPECT_THAT(GetValidatedCompression("dataset_id", metadata),
              IsOkAndHolds(DataServiceMetadata::COMPRESSION_SNAPPY));
}

TEST(UtilsTest, InvalidCompression) {
  DataServiceMetadata metadata;
  EXPECT_THAT(GetValidatedCompression("dataset_id", metadata),
              StatusIs(error::INTERNAL));
}

TEST(UtilsTest, EstimateCardinalityEmptyDataset) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  DataServiceMetadata metadata;
  metadata.set_cardinality(0);
  EXPECT_EQ(EstimateCardinality(processing_mode, metadata,
                                /*is_coordinated_read=*/false),
            0);
}

TEST(UtilsTest, EstimateCardinalityInfiniteDataset) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  DataServiceMetadata metadata;
  metadata.set_cardinality(kInfiniteCardinality);
  EXPECT_EQ(EstimateCardinality(processing_mode, metadata,
                                /*is_coordinated_read=*/false),
            kInfiniteCardinality);

  processing_mode.set_sharding_policy(ProcessingModeDef::DYNAMIC);
  EXPECT_EQ(EstimateCardinality(processing_mode, metadata,
                                /*is_coordinated_read=*/false),
            kInfiniteCardinality);
}

TEST(UtilsTest, EstimateCardinalityCoordinatedRead) {
  ProcessingModeDef processing_mode;
  DataServiceMetadata metadata;
  EXPECT_EQ(EstimateCardinality(processing_mode, metadata,
                                /*is_coordinated_read=*/true),
            kInfiniteCardinality);
}

TEST(UtilsTest, EstimateCardinalityUnknownCardinality) {
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  DataServiceMetadata metadata;
  metadata.set_cardinality(10);
  EXPECT_EQ(EstimateCardinality(processing_mode, metadata,
                                /*is_coordinated_read=*/false),
            kUnknownCardinality);
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
