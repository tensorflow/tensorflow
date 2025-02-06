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

#ifndef XLA_TSL_PLATFORM_CLOUD_OAUTH_CLIENT_H_
#define XLA_TSL_PLATFORM_CLOUD_OAUTH_CLIENT_H_

#include <memory>

#include "json/json.h"
#include "xla/tsl/platform/cloud/http_request.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"

namespace tsl {

/// OAuth 2.0 client.
class OAuthClient {
 public:
  OAuthClient();
  explicit OAuthClient(
      std::unique_ptr<HttpRequest::Factory> http_request_factory, Env* env);
  virtual ~OAuthClient() {}

  /// \brief Retrieves a bearer token using a private key.
  ///
  /// Retrieves the authentication bearer token using a JSON file
  /// with the client's private key.
  virtual absl::Status GetTokenFromServiceAccountJson(
      Json::Value json, absl::string_view oauth_server_uri,
      absl::string_view scope, string* token, uint64* expiration_timestamp_sec);

  /// Retrieves a bearer token using a refresh token.
  virtual absl::Status GetTokenFromRefreshTokenJson(
      Json::Value json, absl::string_view oauth_server_uri, string* token,
      uint64* expiration_timestamp_sec);

  /// Parses the JSON response with the token from an OAuth 2.0 server.
  virtual absl::Status ParseOAuthResponse(absl::string_view response,
                                          uint64 request_timestamp_sec,
                                          string* token,
                                          uint64* expiration_timestamp_sec);

 private:
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;
  Env* env_;
  OAuthClient(const OAuthClient&) = delete;
  void operator=(const OAuthClient&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_CLOUD_OAUTH_CLIENT_H_
