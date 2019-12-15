/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/google_auth_provider.h"

#include <stdlib.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

constexpr char kTestData[] = "core/platform/cloud/testdata/";

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {}

  uint64 NowSeconds() const override { return now; }
  uint64 now = 10000;
};

class FakeOAuthClient : public OAuthClient {
 public:
  Status GetTokenFromServiceAccountJson(
      Json::Value json, StringPiece oauth_server_uri, StringPiece scope,
      string* token, uint64* expiration_timestamp_sec) override {
    provided_credentials_json = json;
    *token = return_token;
    *expiration_timestamp_sec = return_expiration_timestamp;
    return Status::OK();
  }

  /// Retrieves a bearer token using a refresh token.
  Status GetTokenFromRefreshTokenJson(
      Json::Value json, StringPiece oauth_server_uri, string* token,
      uint64* expiration_timestamp_sec) override {
    provided_credentials_json = json;
    *token = return_token;
    *expiration_timestamp_sec = return_expiration_timestamp;
    return Status::OK();
  }

  string return_token;
  uint64 return_expiration_timestamp;
  Json::Value provided_credentials_json;
};

}  // namespace

class GoogleAuthProviderTest : public ::testing::Test {
 protected:
  void SetUp() override { ClearEnvVars(); }

  void TearDown() override { ClearEnvVars(); }

  void ClearEnvVars() {
    unsetenv("CLOUDSDK_CONFIG");
    unsetenv("GOOGLE_APPLICATION_CREDENTIALS");
    unsetenv("GOOGLE_AUTH_TOKEN_FOR_TESTING");
    unsetenv("NO_GCE_CHECK");
  }
};

TEST_F(GoogleAuthProviderTest, EnvironmentVariable_Caching) {
  setenv("GOOGLE_APPLICATION_CREDENTIALS",
         io::JoinPath(
             io::JoinPath(testing::TensorFlowSrcRoot(), kTestData).c_str(),
             "service_account_credentials.json")
             .c_str(),
         1);
  setenv("CLOUDSDK_CONFIG",
         io::JoinPath(testing::TensorFlowSrcRoot(), kTestData).c_str(),
         1);  // Will not be used.

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests;

  FakeEnv env;

  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);
  oauth_client->return_token = "fake-token";
  oauth_client->return_expiration_timestamp = env.NowSeconds() + 3600;

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);
  EXPECT_EQ("fake_key_id",
            oauth_client->provided_credentials_json.get("private_key_id", "")
                .asString());

  // Check that the token is re-used if not expired.
  oauth_client->return_token = "new-fake-token";
  env.now += 3000;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);

  // Check that the token is re-generated when almost expired.
  env.now += 598;  // 2 seconds before expiration
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("new-fake-token", token);
}

TEST_F(GoogleAuthProviderTest, GCloudRefreshToken) {
  setenv("CLOUDSDK_CONFIG",
         io::JoinPath(testing::TensorFlowSrcRoot(), kTestData).c_str(), 1);

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests;

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));

  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);
  oauth_client->return_token = "fake-token";
  oauth_client->return_expiration_timestamp = env.NowSeconds() + 3600;

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);
  EXPECT_EQ("fake-refresh-token",
            oauth_client->provided_credentials_json.get("refresh_token", "")
                .asString());
}

TEST_F(GoogleAuthProviderTest, RunningOnGCE) {
  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           R"(
          {
            "access_token":"fake-gce-token",
            "expires_in": 3920,
            "token_type":"Bearer"
          })"),
       // The first token refresh request fails and will be retried.
       new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           R"(
              {
                "access_token":"new-fake-gce-token",
                "expires_in": 3920,
                "token_type":"Bearer"
              })")});

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-gce-token", token);

  // Check that the token is re-used if not expired.
  env.now += 3700;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-gce-token", token);

  // Check that the token is re-generated when almost expired.
  env.now += 598;  // 2 seconds before expiration
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("new-fake-gce-token", token);
}

TEST_F(GoogleAuthProviderTest, OverrideForTesting) {
  setenv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "tokenForTesting", 1);

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> empty_requests;
  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&empty_requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("tokenForTesting", token);
}

TEST_F(GoogleAuthProviderTest, NothingAvailable) {
  auto oauth_client = new FakeOAuthClient;

  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
      "/default/token\n"
      "Header Metadata-Flavor: Google\n",
      "", errors::NotFound("404"), 404)});

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);
}

TEST_F(GoogleAuthProviderTest, NoGceCheckEnvironmentVariable) {
  setenv("NO_GCE_CHECK", "True", 1);
  auto oauth_client = new FakeOAuthClient;

  FakeEnv env;
  // If the env var above isn't respected, attempting to fetch a token
  // from GCE will segfault (as the metadata client is null).
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              nullptr, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);

  // We confirm that our env var is case insensitive.
  setenv("NO_GCE_CHECK", "true", 1);
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);

  // We also want to confirm that our empty token has a short expiration set: we
  // now set a testing token, and confirm that it's returned instead of our
  // empty token.
  setenv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "newToken", 1);
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("newToken", token);
}

}  // namespace tensorflow
