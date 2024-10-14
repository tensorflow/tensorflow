/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

absl::Status CheckVersions(const VersionDef& versions, int consumer,
                           int min_producer, const char* upper_name,
                           const char* lower_name) {
  // Guard against the caller misordering the arguments
  if (consumer < min_producer) {
    return errors::Internal(upper_name, " version check has consumer ",
                            consumer, " < min_producer ", min_producer, ".");
  }

  // Check versions
  if (versions.producer() < min_producer) {
    return errors::InvalidArgument(
        upper_name, " producer version ", versions.producer(),
        " below min producer ", min_producer, " supported by TensorFlow ",
        TF_VERSION_STRING, ".  Please regenerate your ", lower_name, ".");
  }
  if (versions.min_consumer() > consumer) {
    return errors::InvalidArgument(
        upper_name, " min consumer version ", versions.min_consumer(),
        " above current version ", consumer, " for TensorFlow ",
        TF_VERSION_STRING, ".  Please upgrade TensorFlow.");
  }
  for (const int bad_consumer : versions.bad_consumers()) {
    if (bad_consumer == consumer) {
      return errors::InvalidArgument(
          upper_name, " disallows consumer version ", bad_consumer,
          ".  Please upgrade TensorFlow: this version is likely buggy.");
    }
  }

  // All good!
  return absl::OkStatus();
}

}  // namespace tensorflow
