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

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool CanUseCudnn() {
  const char* tf_use_cudnn = getenv("TF_USE_CUDNN");
  if (tf_use_cudnn != nullptr) {
    string tf_use_cudnn_str = tf_use_cudnn;
    if (tf_use_cudnn_str == "0") {
      return false;
    }
  }
  return true;
}

}  // namespace tensorflow
