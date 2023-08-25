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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_

#include <memory>

#include "tensorflow/core/platform/cloud/auth_provider.h"
#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"
#include "tensorflow/core/platform/cloud/oauth_client.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/tsl/platform/cloud/google_auth_provider.h"

namespace tensorflow {
using tsl::GoogleAuthProvider;  // NOLINT(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GOOGLE_AUTH_PROVIDER_H_
