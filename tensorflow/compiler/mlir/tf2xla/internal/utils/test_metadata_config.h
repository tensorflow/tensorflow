/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_TEST_METADATA_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_TEST_METADATA_CONFIG_H_

#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Fills in arg_shapes and metadata_proto with appropriate values based on the
// input mlir module.
absl::Status ConfigureMetadata(absl::string_view mlir_module_str,
                               std::vector<TensorShape>& arg_shapes,
                               tpu::TPUCompileMetadataProto& metadata_proto);

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_UTILS_TEST_METADATA_CONFIG_H_
