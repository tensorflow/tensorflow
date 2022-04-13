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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_PROTO_REWRITE_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_PROTO_REWRITE_H_

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {

// Validates the TPU embedding configuration has been populated correctly and
// fills in missing fields. The user model is expected to fill in exactly one of
// the following:
//
// (1) batch_size_per_tensor_core and TableDescriptor.num_features, or
// (2) feature_descriptor.
//
// (1) If the user model fills in batch_size_per_tensor_core and
// TableDescriptor.num_features, this function validates that the
// feature_descriptor has not been filled in, and then populates
// feature_descriptor with appropriate values.
//
// (2) If the user model fills in feature_descriptor, this function validates
// that batch_size_per_tensor_core and TableDescriptor.num_features have not
// been filled in, and then populated them with appropriate values.
Status PopulateMissingFieldsInTPUEmbeddingConfig(
    tpu::TPUEmbeddingConfiguration* config);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_CONFIGURATION_PROTO_REWRITE_H_
