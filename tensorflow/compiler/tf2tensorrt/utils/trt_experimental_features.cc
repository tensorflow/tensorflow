/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace tensorrt {

bool isExperimentalFeatureActivated(string feature_name) {
  string envvar_str;
  TF_CHECK_OK(
      ReadStringFromEnvVar("TF_TRT_EXPERIMENTAL_FEATURES", "", &envvar_str));
  return envvar_str.find(feature_name) != string::npos;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
