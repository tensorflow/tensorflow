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

#include "tensorflow/core/platform/cuda_libdevice_path.h"

#include <stdlib.h>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

string CudaRoot() {
  // 'bazel test' sets TEST_SRCDIR.
  const string kRelativeCudaRoot = io::JoinPath("local_config_cuda", "cuda");
  const char* test_srcdir = getenv("TEST_SRCDIR");
  if (test_srcdir && test_srcdir[0] != '\0') {
    return io::JoinPath(test_srcdir, kRelativeCudaRoot);
  }

  LOG(INFO) << "TEST_SRCDIR environment variable not set: using "
            << kRelativeCudaRoot
            << " under this executable's runfiles directory as the CUDA root.";
  return io::JoinPath(
      strings::StrCat(Env::Default()->GetExecutablePath(), ".runfiles"),
      kRelativeCudaRoot);
}

}  // namespace tensorflow
