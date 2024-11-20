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
#include "xla/tsl/platform/cloud/oauth_client.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
using tsl::OAuthClient;  // NOLINT(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_OAUTH_CLIENT_H_
