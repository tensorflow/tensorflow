/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_ZONE_PROVIDER_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_ZONE_PROVIDER_H_

#include <string>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

/// Interface for a provider of cloud instance zone
class ZoneProvider {
 public:
  virtual ~ZoneProvider() {}

  /// \brief  Gets the zone of the Cloud instance and set the result in `zone`.
  /// Returns OK if success.
  ///
  /// Returns an empty string in the case where the zone does not match the
  /// expected format
  /// Safe for concurrent use by multiple threads.
  virtual Status GetZone(string* zone) = 0;

  static Status GetZone(ZoneProvider* provider, string* zone) {
    if (!provider) {
      return errors::Internal("Zone provider is required.");
    }
    return provider->GetZone(zone);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_ZONE_PROVIDER_H_
