/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/cloud/compute_engine_zone_provider.h"

#include "tsl/platform/cloud/http_request_fake.h"
#include "tsl/platform/test.h"

namespace tsl {

class ComputeEngineZoneProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(ComputeEngineZoneProviderTest, GetZone) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: http://metadata.google.internal/computeMetadata/v1/instance/zone\n"
      "Header Metadata-Flavor: Google\n",
      "projects/123456789/zones/us-west1-b")});

  auto httpRequestFactory = std::make_shared<FakeHttpRequestFactory>(&requests);

  auto metadata_client = std::make_shared<ComputeEngineMetadataClient>(
      httpRequestFactory, RetryConfig(0 /* init_delay_time_us */));

  ComputeEngineZoneProvider provider(metadata_client);

  string zone;

  TF_EXPECT_OK(provider.GetZone(&zone));
  EXPECT_EQ("us-west1-b", zone);
  // Test caching, should be no further requests
  TF_EXPECT_OK(provider.GetZone(&zone));
}

TEST_F(ComputeEngineZoneProviderTest, InvalidZoneString) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: http://metadata.google.internal/computeMetadata/v1/instance/zone\n"
      "Header Metadata-Flavor: Google\n",
      "invalidresponse")});

  auto httpRequestFactory = std::make_shared<FakeHttpRequestFactory>(&requests);

  auto metadata_client = std::make_shared<ComputeEngineMetadataClient>(
      httpRequestFactory, RetryConfig(0 /* init_delay_time_us */));

  ComputeEngineZoneProvider provider(metadata_client);

  string zone;

  TF_EXPECT_OK(provider.GetZone(&zone));
  EXPECT_EQ("", zone);
}

}  // namespace tsl
