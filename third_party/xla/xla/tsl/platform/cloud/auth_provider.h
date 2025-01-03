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

#ifndef XLA_TSL_PLATFORM_CLOUD_AUTH_PROVIDER_H_
#define XLA_TSL_PLATFORM_CLOUD_AUTH_PROVIDER_H_

#include <string>

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"

namespace tsl {

/// Interface for a provider of authentication bearer tokens.
class AuthProvider {
 public:
  virtual ~AuthProvider() {}

  /// \brief Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual absl::Status GetToken(string* t) = 0;

  static absl::Status GetToken(AuthProvider* provider, string* token) {
    if (!provider) {
      return errors::Internal("Auth provider is required.");
    }
    return provider->GetToken(token);
  }
};

/// No-op auth provider, which will only work for public objects.
class EmptyAuthProvider : public AuthProvider {
 public:
  absl::Status GetToken(string* token) override {
    *token = "";
    return absl::OkStatus();
  }
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_CLOUD_AUTH_PROVIDER_H_
