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

#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {

namespace {
mutex* get_lock() {
  static mutex lock(LINKER_INITIALIZED);
  return &lock;
}

using CredentialsFactories =
    std::unordered_map<std::string, CredentialsFactory*>;
CredentialsFactories& credentials_factories() {
  static auto& factories = *new CredentialsFactories();
  return factories;
}
}  // namespace

void CredentialsFactory::Register(CredentialsFactory* factory) {
  mutex_lock l(*get_lock());
  if (!credentials_factories().insert({factory->Protocol(), factory}).second) {
    LOG(ERROR)
        << "Two credentials factories are being registered with protocol "
        << factory->Protocol() << ". Which one gets used is undefined.";
  }
}

Status CredentialsFactory::Get(absl::string_view protocol,
                               CredentialsFactory** out) {
  mutex_lock l(*get_lock());
  auto it = credentials_factories().find(std::string(protocol));
  if (it != credentials_factories().end()) {
    *out = it->second;
    return Status::OK();
  }

  std::vector<string> available_types;
  for (const auto& factory : credentials_factories()) {
    available_types.push_back(factory.first);
  }

  return errors::NotFound("No credentials factory has been registered for ",
                          "protocol ", protocol,
                          ". The available types are: [ ",
                          absl::StrJoin(available_types, ", "), " ]");
}

Status CredentialsFactory::CreateServerCredentials(
    absl::string_view protocol,
    std::shared_ptr<::grpc::ServerCredentials>* out) {
  CredentialsFactory* factory;
  TF_RETURN_IF_ERROR(CredentialsFactory::Get(protocol, &factory));
  TF_RETURN_IF_ERROR(factory->CreateServerCredentials(out));
  return Status::OK();
}

Status CredentialsFactory::CreateClientCredentials(
    absl::string_view protocol,
    std::shared_ptr<::grpc::ChannelCredentials>* out) {
  CredentialsFactory* factory;
  TF_RETURN_IF_ERROR(CredentialsFactory::Get(protocol, &factory));
  TF_RETURN_IF_ERROR(factory->CreateClientCredentials(out));
  return Status::OK();
}

bool CredentialsFactory::Exists(absl::string_view protocol) {
  mutex_lock l(*get_lock());
  return credentials_factories().find(std::string(protocol)) !=
         credentials_factories().end();
}

class InsecureCredentialsFactory : public CredentialsFactory {
 public:
  std::string Protocol() override { return "grpc"; }

  Status CreateServerCredentials(
      std::shared_ptr<::grpc::ServerCredentials>* out) override {
    *out = ::grpc::InsecureServerCredentials();
    return Status::OK();
  }

  Status CreateClientCredentials(
      std::shared_ptr<::grpc::ChannelCredentials>* out) override {
    *out = ::grpc::InsecureChannelCredentials();
    return Status::OK();
  }
};

class InsecureCredentialsRegistrar {
 public:
  InsecureCredentialsRegistrar() {
    auto factory = new InsecureCredentialsFactory();
    CredentialsFactory::Register(factory);
  }
};
static InsecureCredentialsRegistrar registrar;

}  // namespace data
}  // namespace tensorflow
