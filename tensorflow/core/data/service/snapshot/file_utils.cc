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
#include "tensorflow/core/data/service/snapshot/file_utils.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {

tsl::Status AtomicallyWriteStringToFile(absl::string_view filename,
                                        absl::string_view str, tsl::Env* env) {
  std::string uncommitted_filename;
  if (!env->LocalTempFilename(&uncommitted_filename)) {
    return tsl::errors::Internal("Failed to write file at ", filename,
                                 ". Requested to write string: ", str);
  }

  TF_RETURN_IF_ERROR(WriteStringToFile(env, uncommitted_filename, str));
  return env->RenameFile(uncommitted_filename, std::string(filename));
}

}  // namespace data
}  // namespace tensorflow
