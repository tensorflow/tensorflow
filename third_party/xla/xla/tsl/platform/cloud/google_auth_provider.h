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

#ifndef XLA_TSL_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_
#define XLA_TSL_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_

#include <memory>

#include "xla/tsl/platform/cloud/auth_provider.h"
#include "xla/tsl/platform/cloud/compute_engine_metadata_client.h"
#include "xla/tsl/platform/cloud/oauth_client.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {

/// Implementation based on Google Application Default Credentials.
class GoogleAuthProvider : public AuthProvider {
 public:
  GoogleAuthProvider(std::shared_ptr<ComputeEngineMetadataClient>
                         compute_engine_metadata_client);
  explicit GoogleAuthProvider(std::unique_ptr<OAuthClient> oauth_client,
                              std::shared_ptr<ComputeEngineMetadataClient>
                                  compute_engine_metadata_client,
                              Env* env);
  virtual ~GoogleAuthProvider() {}

  /// \brief Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  absl::Status GetToken(string* token) override;

 private:
  /// \brief Gets the bearer token from files.
  ///
  /// Tries the file from $GOOGLE_APPLICATION_CREDENTIALS and the
  /// standard gcloud tool's location.
  absl::Status GetTokenFromFiles() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Gets the bearer token from Google Compute Engine environment.
  absl::Status GetTokenFromGce() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Gets the bearer token from the system env variable, for testing purposes.
  absl::Status GetTokenForTesting() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::unique_ptr<OAuthClient> oauth_client_;
  std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client_;
  Env* env_;
  mutex mu_;
  string current_token_ TF_GUARDED_BY(mu_);
  uint64 expiration_timestamp_sec_ TF_GUARDED_BY(mu_) = 0;
  GoogleAuthProvider(const GoogleAuthProvider&) = delete;
  void operator=(const GoogleAuthProvider&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_
