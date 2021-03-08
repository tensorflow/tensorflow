/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

static Status InitGraphModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();
  TF_RETURN_IF_ERROR(grappler::InitGraphPlugin(dso_handle));
  return Status::OK();
}

Status RegisterPluggableDevicePlugin(void* dso_handle) {
  // Step1 Init Graph Module
  TF_RETURN_IF_ERROR(InitGraphModule(dso_handle));

  // Step2 Init Device/Kernel Module

  return Status::OK();
}

}  // namespace tensorflow
