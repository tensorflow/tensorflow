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

#include <signal.h>

#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace testing {

int PickUnusedPortOrDie() { return internal::PickUnusedPortOrDie(); }

string TensorFlowSrcRoot() {
  // 'bazel test' sets TEST_SRCDIR, and also TEST_WORKSPACE if a new
  // enough version of bazel is used.
  const char* env = getenv("TEST_SRCDIR");
  const char* workspace = getenv("TEST_WORKSPACE");
  if (env && env[0] != '\0') {
    if (workspace && workspace[0] != '\0') {
      return strings::StrCat(env, "/", workspace, "/tensorflow");
    } else {
      return strings::StrCat(env, "/tensorflow");
    }
  } else {
    LOG(WARNING) << "TEST_SRCDIR environment variable not set: "
                 << "using $PWD/tensorflow as TensorFlowSrcRoot() for tests.";
    return "tensorflow";
  }
}

}  // namespace testing
}  // namespace tensorflow
