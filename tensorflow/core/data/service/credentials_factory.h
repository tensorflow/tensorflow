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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_CREDENTIALS_FACTORY_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CREDENTIALS_FACTORY_H_

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Credential factory implementations should be threadsafe since all callers
// to `GetCredentials` will get the same instance of `CredentialsFactory`.
class CredentialsFactory {
 public:
  virtual ~CredentialsFactory() = default;

  // Returns a protocol name for the credentials factory. This is the string to
  // look up with `GetCredentials` to find the registered credentials factory.
  virtual std::string Protocol() = 0;

  // Stores server credentials to `*out`.
  virtual Status CreateServerCredentials(
      std::shared_ptr<::grpc::ServerCredentials>* out) = 0;

  // Stores client credentials to `*out`.
  virtual Status CreateClientCredentials(
      std::shared_ptr<::grpc::ChannelCredentials>* out) = 0;

  // Registers a credentials factory.
  static void Register(CredentialsFactory* factory);

  // Creates server credentials using the credentials factory registered as
  // `protocol`, and stores them to `*out`.
  static Status CreateServerCredentials(
      absl::string_view protocol,
      std::shared_ptr<::grpc::ServerCredentials>* out);

  // Creates client credentials using the credentials factory registered as
  // `protocol`, and stores them to `*out`.
  static Status CreateClientCredentials(
      absl::string_view protocol,
      std::shared_ptr<::grpc::ChannelCredentials>* out);

  // Returns whether a factory has been registered under the given protocl name.
  static bool Exists(absl::string_view protocol);

 private:
  // Gets the credentials factory registered via `Register` for the specified
  // protocol, and stores it to `*out`.
  static Status Get(const absl::string_view protocol, CredentialsFactory** out);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CREDENTIALS_FACTORY_H_
