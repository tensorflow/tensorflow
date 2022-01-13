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

#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"

#include "tensorflow/core/platform/status.h"

namespace tensorflow {

Status PopulateEmbeddingFeatureDescriptor(
    tpu::TPUEmbeddingConfiguration& tpu_embedding_config) {
  // Return early if the feature descriptor is already set.
  if (tpu_embedding_config.feature_descriptor_size() != 0) {
    return Status::OK();
  }
  for (int i = 0; i < tpu_embedding_config.table_descriptor_size(); i++) {
    for (int j = 0; j < tpu_embedding_config.table_descriptor(i).num_features();
         j++) {
      tpu::TPUEmbeddingConfiguration::FeatureDescriptor* feature_descriptor =
          tpu_embedding_config.add_feature_descriptor();
      feature_descriptor->set_table_id(i);
      feature_descriptor->add_input_shape(
          tpu_embedding_config.batch_size_per_tensor_core());
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
