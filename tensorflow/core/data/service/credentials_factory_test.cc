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
#include "tensorflow/core/data/service/credentials_factory.h"

#include <memory>
#include <string>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
constexpr char kFailedToCreateServerCredentials[] =
    "Failed to create server credentials.";
constexpr char kFailedToCreateClientCredentials[] =
    "Failed to create client credentials.";

class TestCredentialsFactory : public CredentialsFactory {
 public:
  std::string Protocol() override { return "test"; }

  absl::Status CreateServerCredentials(
      std::shared_ptr<grpc::ServerCredentials>* out) override {
    return errors::Internal(kFailedToCreateServerCredentials);
  }

  absl::Status CreateClientCredentials(
      std::shared_ptr<grpc::ChannelCredentials>* out) override {
    return errors::Internal(kFailedToCreateClientCredentials);
  }
};
}  // namespace

TEST(CredentialsFactory, Register) {
  TestCredentialsFactory test_factory;
  CredentialsFactory::Register(&test_factory);
  std::shared_ptr<grpc::ServerCredentials> server_credentials;
  ASSERT_EQ(errors::Internal(kFailedToCreateServerCredentials),
            CredentialsFactory::CreateServerCredentials(test_factory.Protocol(),
                                                        &server_credentials));
  std::shared_ptr<grpc::ChannelCredentials> client_credentials;
  ASSERT_EQ(errors::Internal(kFailedToCreateClientCredentials),
            CredentialsFactory::CreateClientCredentials(test_factory.Protocol(),
                                                        &client_credentials));
}

TEST(CredentialsFactory, DefaultGrpcProtocol) {
  std::shared_ptr<grpc::ServerCredentials> server_credentials;
  TF_ASSERT_OK(
      CredentialsFactory::CreateServerCredentials("grpc", &server_credentials));
  std::shared_ptr<grpc::ChannelCredentials> client_credentials;
  TF_ASSERT_OK(
      CredentialsFactory::CreateClientCredentials("grpc", &client_credentials));
}

TEST(CredentialsFactory, MissingServerProtocol) {
  std::shared_ptr<grpc::ServerCredentials> server_credentials;
  absl::Status s = CredentialsFactory::CreateServerCredentials(
      "unknown_protocol", &server_credentials);
  ASSERT_EQ(error::Code::NOT_FOUND, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.ToString(),
                        "No credentials factory has been registered for "
                        "protocol unknown_protocol"));
}

TEST(CredentialsFactory, MissingClientProtocol) {
  std::shared_ptr<grpc::ChannelCredentials> client_credentials;
  absl::Status s = CredentialsFactory::CreateClientCredentials(
      "unknown_protocol", &client_credentials);
  ASSERT_EQ(error::Code::NOT_FOUND, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.ToString(),
                        "No credentials factory has been registered for "
                        "protocol unknown_protocol"));
}

}  // namespace data
}  // namespace tensorflow
