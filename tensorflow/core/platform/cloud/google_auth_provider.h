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

#ifndef TENSORFLOW_CORE_PLATFORM_GOOGLE_AUTH_PROVIDER_H_
#define TENSORFLOW_CORE_PLATFORM_GOOGLE_AUTH_PROVIDER_H_

#include <memory>
#include "tensorflow/core/platform/cloud/auth_provider.h"
#include "tensorflow/core/platform/cloud/oauth_client.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

/// Implementation based on Google Application Default Credentials.
class GoogleAuthProvider : public AuthProvider {
 public:
  GoogleAuthProvider();
  explicit GoogleAuthProvider(
      std::unique_ptr<OAuthClient> oauth_client,
      std::unique_ptr<HttpRequest::Factory> http_request_factory, Env* env);
  virtual ~GoogleAuthProvider() {}

  /// \brief Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  Status GetToken(string* token) override;

 private:
  /// \brief Gets the bearer token from files.
  ///
  /// Tries the file from $GOOGLE_APPLICATION_CREDENTIALS and the
  /// standard gcloud tool's location.
  Status GetTokenFromFiles();

  /// Gets the bearer token from Google Compute Engine environment.
  Status GetTokenFromGce();

  std::unique_ptr<OAuthClient> oauth_client_;
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;
  Env* env_;
  mutex mu_;
  string current_token_ GUARDED_BY(mu_);
  uint64 expiration_timestamp_sec_ GUARDED_BY(mu_) = 0;
  TF_DISALLOW_COPY_AND_ASSIGN(GoogleAuthProvider);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_GOOGLE_AUTH_PROVIDER_H_
