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

#include <cstdint>
#include <set>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

namespace tensorflow {
namespace tpu {

absl::StatusOr<int32_t> ComputeTotalTagCountForOptimizerDynamicInputs(
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config) {
  // Ordering of tag elements helps make the subsequent error checking simpler.
  std::set<int32_t> tag_set;
  for (const auto& table_descriptor : tpu_embedding_config.table_descriptor()) {
    const auto& opt_params = table_descriptor.optimization_parameters();
    const auto tags_for_table = GetOptimizerDynamicInputTags(opt_params);
    tag_set.insert(tags_for_table.begin(), tags_for_table.end());
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

  return static_cast<int32_t>(tag_set.size());
}

}  // namespace tpu
}  // namespace tensorflow
