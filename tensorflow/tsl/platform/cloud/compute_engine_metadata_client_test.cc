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

#include "tensorflow/tsl/platform/cloud/compute_engine_metadata_client.h"

#include "tensorflow/tsl/platform/cloud/http_request_fake.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {

class ComputeEngineMetadataClientTest : public ::testing::Test {
 protected:
  void SetUp() override { ClearEnvVars(); }

  void TearDown() override { ClearEnvVars(); }

  void ClearEnvVars() { unsetenv("GCE_METADATA_HOST"); }
};

TEST_F(ComputeEngineMetadataClientTest, GetMetadata) {
  const string example_response = "example response";

  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: http://metadata.google.internal/computeMetadata/v1/instance"
      "/service-accounts/default/token\n"
      "Header Metadata-Flavor: Google\n",
      example_response)});

  std::shared_ptr<HttpRequest::Factory> http_factory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  ComputeEngineMetadataClient client(http_factory,
                                     RetryConfig(0 /* init_delay_time_us */));

  std::vector<char> result;
  TF_EXPECT_OK(
      client.GetMetadata("instance/service-accounts/default/token", &result));
  std::vector<char> expected(example_response.begin(), example_response.end());
  EXPECT_EQ(expected, result);
}

TEST_F(ComputeEngineMetadataClientTest, GetCustomMetadataEndpoint) {
  const string example_response = "example response";
  setenv("GCE_METADATA_HOST", "foo.bar", 1);

  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest("Uri: http://foo.bar/computeMetadata/v1/instance"
                           "/service-accounts/default/token\n"
                           "Header Metadata-Flavor: Google\n",
                           example_response)});

  std::shared_ptr<HttpRequest::Factory> http_factory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  ComputeEngineMetadataClient client(http_factory,
                                     RetryConfig(0 /* init_delay_time_us */));

  std::vector<char> result;
  TF_EXPECT_OK(
      client.GetMetadata("instance/service-accounts/default/token", &result));
  std::vector<char> expected(example_response.begin(), example_response.end());
  EXPECT_EQ(expected, result);
}

TEST_F(ComputeEngineMetadataClientTest, RetryOnFailure) {
  const string example_response = "example response";

  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: http://metadata.google.internal/computeMetadata/v1/instance"
           "/service-accounts/default/token\n"
           "Header Metadata-Flavor: Google\n",
           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest(
           "Uri: http://metadata.google.internal/computeMetadata/v1/instance"
           "/service-accounts/default/token\n"
           "Header Metadata-Flavor: Google\n",
           example_response)});

  std::shared_ptr<HttpRequest::Factory> http_factory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  ComputeEngineMetadataClient client(http_factory,
                                     RetryConfig(0 /* init_delay_time_us */));

  std::vector<char> result;
  TF_EXPECT_OK(
      client.GetMetadata("instance/service-accounts/default/token", &result));
  std::vector<char> expected(example_response.begin(), example_response.end());
  EXPECT_EQ(expected, result);
}

}  // namespace tsl
