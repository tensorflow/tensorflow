/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_embedding_configuration_utils.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"

namespace tensorflow {
namespace tpu {

absl::StatusOr<int32_t> ComputeTotalTagCountForDynamicLearningRates(
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config) {
  // Ordering of tag elements helps make the subsequent error checking simpler.
  std::set<int32_t> tag_set;

  for (const auto& table_descriptor : tpu_embedding_config.table_descriptor()) {
    const auto& lr_spec =
        table_descriptor.optimization_parameters().learning_rate();
    if (lr_spec.has_dynamic()) {
      tag_set.insert(lr_spec.dynamic().tag());
    }
  }

  // Traverse the tag set to determine that tags are contiguous.
  int32_t next_tag = 0;
  for (const int32_t tag : tag_set) {
    if (tag != next_tag) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dynamic learning rate tag: %d not found in the TPU embedding "
          "configuration, instead found: %d. tag set size: %d",
          next_tag, tag, tag_set.size()));
    }
    ++next_tag;
  }

  return tag_set.size();
}

}  // namespace tpu
}  // namespace tensorflow
