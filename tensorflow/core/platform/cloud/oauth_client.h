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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_OAUTH_CLIENT_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_OAUTH_CLIENT_H_

#include <memory>

#include "json/json.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

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
  virtual Status GetTokenFromServiceAccountJson(
      Json::Value json, StringPiece oauth_server_uri, StringPiece scope,
      string* token, uint64* expiration_timestamp_sec);

  /// Retrieves a bearer token using a refresh token.
  virtual Status GetTokenFromRefreshTokenJson(Json::Value json,
                                              StringPiece oauth_server_uri,
                                              string* token,
                                              uint64* expiration_timestamp_sec);

  /// Parses the JSON response with the token from an OAuth 2.0 server.
  virtual Status ParseOAuthResponse(StringPiece response,
                                    uint64 request_timestamp_sec, string* token,
                                    uint64* expiration_timestamp_sec);

 private:
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;
  Env* env_;
  TF_DISALLOW_COPY_AND_ASSIGN(OAuthClient);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_OAUTH_CLIENT_H_
