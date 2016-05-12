/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/use_cudnn.h"

#include <stdlib.h>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

static bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val) {
  const char* tf_env_var_val = getenv(env_var_name);
  if (tf_env_var_val != nullptr) {
    StringPiece tf_env_var_val_str(tf_env_var_val);
    if (tf_env_var_val_str == "0") {
      return false;
    }
    return true;
  }
  return default_val;
}

bool CanUseCudnn() { return ReadBoolFromEnvVar("TF_USE_CUDNN", true); }

bool CudnnUseAutotune() {
  return ReadBoolFromEnvVar("TF_CUDNN_USE_AUTOTUNE", false);
}

}  // namespace tensorflow
