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
#ifndef TENSORFLOW_CORE_TFRT_MLA_MLA_UTILS_H_
#define TENSORFLOW_CORE_TFRT_MLA_MLA_UTILS_H_

// This file contains stub implementations for Google internal MLA APIs.

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tfrt_stub {

inline std::string CopySavedModelFromTestDataToTempDir(
    absl::string_view tf_dir, absl::string_view saved_model_name) {
  return "";
}

inline Status ConvertSavedModelAndAddToMla(
    absl::string_view saved_model_path, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    const std::vector<std::string>& entry_points,
    absl::string_view mla_module_name) {
  return tensorflow::errors::Unimplemented("Not supported in OSS");
}

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_MLA_MLA_UTILS_H_
