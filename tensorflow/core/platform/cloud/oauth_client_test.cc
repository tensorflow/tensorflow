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

#include "tensorflow/core/platform/cloud/oauth_client.h"

#include <fstream>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestData[] = "core/platform/cloud/testdata/";

constexpr char kTokenJson[] = R"(
    {
      "access_token":"1/fFAGRNJru1FTz70BzhT3Zg",
      "expires_in":3920,
      "token_type":"Bearer"
    })";

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {}

  uint64 NowSeconds() const override { return now; }
  uint64 now = 10000;
};

}  // namespace

TEST(OAuthClientTest, ParseOAuthResponse) {
  const uint64 request_timestamp = 100;
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(OAuthClient().ParseOAuthResponse(kTokenJson, request_timestamp,
                                                &token, &expiration_timestamp));
  EXPECT_EQ("1/fFAGRNJru1FTz70BzhT3Zg", token);
  EXPECT_EQ(4020, expiration_timestamp);
}

TEST(OAuthClientTest, GetTokenFromRefreshTokenJson) {
  const string credentials_json = R"(
      {
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "refresh_token": "test_refresh_token",
        "type": "authorized_user"
      })";
  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(credentials_json, json));

  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/oauth2/v3/token\n"
      "Post body: client_id=test_client_id&"
      "client_secret=test_client_secret&"
      "refresh_token=test_refresh_token&grant_type=refresh_token\n",
      kTokenJson)});
  FakeEnv env;
  OAuthClient client(std::unique_ptr<HttpRequest::Factory>(
                         new FakeHttpRequestFactory(&requests)),
                     &env);
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(client.GetTokenFromRefreshTokenJson(
      json, "https://www.googleapis.com/oauth2/v3/token", &token,
      &expiration_timestamp));
  EXPECT_EQ("1/fFAGRNJru1FTz70BzhT3Zg", token);
  EXPECT_EQ(13920, expiration_timestamp);
}

TEST(OAuthClientTest, GetTokenFromServiceAccountJson) {
  std::ifstream credentials(
      io::JoinPath(io::JoinPath(testing::TensorFlowSrcRoot(), kTestData),
                   "service_account_credentials.json"));
  ASSERT_TRUE(credentials.is_open());
  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(credentials, json));

  string post_body;
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest("Uri: https://www.googleapis.com/oauth2/v3/token\n",
                           kTokenJson, &post_body)});
  FakeEnv env;
  OAuthClient client(std::unique_ptr<HttpRequest::Factory>(
                         new FakeHttpRequestFactory(&requests)),
                     &env);
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(client.GetTokenFromServiceAccountJson(
      json, "https://www.googleapis.com/oauth2/v3/token",
      "https://test-token-scope.com", &token, &expiration_timestamp));
  EXPECT_EQ("1/fFAGRNJru1FTz70BzhT3Zg", token);
  EXPECT_EQ(13920, expiration_timestamp);

  // Now look at the JWT claim that was sent to the OAuth server.
  StringPiece grant_type, assertion;
  ASSERT_TRUE(strings::Scanner(post_body)
                  .OneLiteral("grant_type=")
                  .RestartCapture()
                  .ScanEscapedUntil('&')
                  .StopCapture()
                  .OneLiteral("&assertion=")
                  .GetResult(&assertion, &grant_type));
  EXPECT_EQ("urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer",
            grant_type);

  int last_dot = assertion.rfind('.');
  string header_dot_claim(assertion.substr(0, last_dot));
  string signature_encoded(assertion.substr(last_dot + 1));

  // Check that 'signature' signs 'header_dot_claim'.

  // Read the serialized public key.
  std::ifstream public_key_stream(
      io::JoinPath(io::JoinPath(testing::TensorFlowSrcRoot(), kTestData),
                   "service_account_public_key.txt"));
  string public_key_serialized(
      (std::istreambuf_iterator<char>(public_key_stream)),
      (std::istreambuf_iterator<char>()));

  // Deserialize the public key.
  auto bio = BIO_new(BIO_s_mem());
  RSA* public_key = nullptr;
  EXPECT_EQ(public_key_serialized.size(),
            BIO_puts(bio, public_key_serialized.c_str()));
  public_key = PEM_read_bio_RSA_PUBKEY(bio, nullptr, nullptr, nullptr);
  EXPECT_TRUE(public_key) << "Could not load the public key from testdata.";

  // Deserialize the signature.
  string signature;
  TF_EXPECT_OK(Base64Decode(signature_encoded, &signature));

  // Actually cryptographically verify the signature.
  const auto md = EVP_sha256();
  auto md_ctx = EVP_MD_CTX_create();
  auto key = EVP_PKEY_new();
  EVP_PKEY_set1_RSA(key, public_key);
  ASSERT_EQ(1, EVP_DigestVerifyInit(md_ctx, nullptr, md, nullptr, key));
  ASSERT_EQ(1, EVP_DigestVerifyUpdate(md_ctx, header_dot_claim.c_str(),
                                      header_dot_claim.size()));
  ASSERT_EQ(1,
            EVP_DigestVerifyFinal(
                md_ctx,
                const_cast<unsigned char*>(
                    reinterpret_cast<const unsigned char*>(signature.data())),
                signature.size()));

  // Free all the crypto-related resources.
  EVP_PKEY_free(key);
  EVP_MD_CTX_destroy(md_ctx);
  RSA_free(public_key);
  BIO_free_all(bio);

  // Now check the content of the header and the claim.
  int dot = header_dot_claim.find_last_of(".");
  string header_encoded = header_dot_claim.substr(0, dot);
  string claim_encoded = header_dot_claim.substr(dot + 1);

  string header, claim;
  TF_EXPECT_OK(Base64Decode(header_encoded, &header));
  TF_EXPECT_OK(Base64Decode(claim_encoded, &claim));

  Json::Value header_json, claim_json;
  EXPECT_TRUE(reader.parse(header, header_json));
  EXPECT_EQ("RS256", header_json.get("alg", Json::Value::null).asString());
  EXPECT_EQ("JWT", header_json.get("typ", Json::Value::null).asString());
  EXPECT_EQ("fake_key_id",
            header_json.get("kid", Json::Value::null).asString());

  EXPECT_TRUE(reader.parse(claim, claim_json));
  EXPECT_EQ("fake-test-project.iam.gserviceaccount.com",
            claim_json.get("iss", Json::Value::null).asString());
  EXPECT_EQ("https://test-token-scope.com",
            claim_json.get("scope", Json::Value::null).asString());
  EXPECT_EQ("https://www.googleapis.com/oauth2/v3/token",
            claim_json.get("aud", Json::Value::null).asString());
  EXPECT_EQ(10000, claim_json.get("iat", Json::Value::null).asInt64());
  EXPECT_EQ(13600, claim_json.get("exp", Json::Value::null).asInt64());
}
}  // namespace tensorflow
