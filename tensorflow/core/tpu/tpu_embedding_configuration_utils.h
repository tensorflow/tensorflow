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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace tpu {

// Returns the total number of unique dynamic input tags used in optimizers. If
// the tag specific is erroneous, returns an invalid argument error. For correct
// tag specification, see the comment next to the OptimizerDynamicInput proto in
// //third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
absl::StatusOr<int32_t> ComputeTotalTagCountForOptimizerDynamicInputs(
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_UTILS_H_
