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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace data {

// Atomically writes `str` to `filename`. Overwrites existing contents if the
// file already exists.
tsl::Status AtomicallyWriteStringToFile(absl::string_view filename,
                                        absl::string_view str, tsl::Env* env);

// Atomically writes the binary representation of `proto` to `filename`.
// Overwrites existing contents if the file already exists.
tsl::Status AtomicallyWriteBinaryProto(absl::string_view filename,
                                       const tsl::protobuf::Message& proto,
                                       tsl::Env* env);

// Atomically writes the text representation of `proto` to `filename`.
// Overwrites existing contents if the file already exists.
tsl::Status AtomicallyWriteTextProto(absl::string_view filename,
                                     const tsl::protobuf::Message& proto,
                                     tsl::Env* env);

// Atomically writes `tensor` to `filename` in TFRecord format. Overwrites
// existing contents if the file already exists.
tsl::Status AtomicallyWriteTFRecord(absl::string_view filename,
                                    const Tensor& tensor,
                                    absl::string_view compression,
                                    tsl::Env* env);

// Returns the relative paths of the children of `directory`, ignoring temporary
// files. Returns an empty vector if the directory does not have any children.
tsl::StatusOr<std::vector<std::string>> GetChildren(absl::string_view directory,
                                                    tsl::Env* env);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SNAPSHOT_FILE_UTILS_H_
