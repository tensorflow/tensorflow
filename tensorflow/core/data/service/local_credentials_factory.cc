/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {
namespace data {

class LocalCredentialsFactory : public CredentialsFactory {
 public:
  std::string Protocol() override { return "grpc+local"; }

  Status CreateServerCredentials(
      std::shared_ptr<::grpc::ServerCredentials>* out) override {
    *out = grpc::experimental::LocalServerCredentials(LOCAL_TCP);
    return Status::OK();
  }

  Status CreateClientCredentials(
      std::shared_ptr<::grpc::ChannelCredentials>* out) override {
    *out = grpc::experimental::LocalCredentials(LOCAL_TCP);
    return Status::OK();
  }
};

class LocalCredentialsRegistrar {
 public:
  LocalCredentialsRegistrar() {
    auto factory = new LocalCredentialsFactory();
    CredentialsFactory::Register(factory);
  }
};
static LocalCredentialsRegistrar registrar;

}  // namespace data
}  // namespace tensorflow
