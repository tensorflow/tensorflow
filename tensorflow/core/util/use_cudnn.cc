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

#include "tensorflow/core/util/use_cudnn.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

bool CanUseCudnn() {
  bool value;
  Status status = ReadBoolFromEnvVar("TF_USE_CUDNN", true, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  return value;
}

bool CudnnUseAutotune() {
  bool value;
  Status status = ReadBoolFromEnvVar("TF_CUDNN_USE_AUTOTUNE", true, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  return value;
}

}  // namespace tensorflow
