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

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace {

// Validates that the batch_size_per_tensor_core and
// TableDescriptor.num_features fields have been populated correctly in the TPU
// embedding configuration.
absl::Status ValidateBatchSizeAndFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config) {
  if (config.batch_size_per_tensor_core() <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid batch_size_per_tensor_core: %d found in the TPU embedding "
        "configuration. Valid values are >0.",
        config.batch_size_per_tensor_core()));
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid num_features: %d found for table: %s in the TPU embedding "
          "configuration. Valid values are >0.",
          table_config.num_features(), table_config.name()));
    }
  }  // table_config
  return absl::OkStatus();
}

// Validates that the batch_size_per_tensor_core and
// TableDescriptor.num_features fields are NOT populated in the TPU embedding
// configuration when the feature descriptor fields are filled in.
absl::Status ValidateBatchSizeAndFeatureCountsAreEmpty(
    const tpu::TPUEmbeddingConfiguration& config) {
  if (config.batch_size_per_tensor_core() != 0) {
    return absl::InvalidArgumentError(
        "Invalid TPU embedding configuration. The batch_size_per_tensor_core "
        "field must NOT be populated when the feature_descriptor fields are "
        "filled in.");
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid TPU embedding configuration. The "
          "TableDescriptor.num_features field must NOT be populated when the "
          "feature_descriptor fields are filled in, num_features is set to %d "
          "for table %s.",
          table_config.num_features(), table_config.name()));
    }
  }  // table_config
  return absl::OkStatus();
}

// Validates that the feature_descriptor fields have been correctly filled in.
// All tables must have at least one input feature.
absl::Status ValidateFeatureDescriptors(
    const tpu::TPUEmbeddingConfiguration& config) {
  const int table_count = config.table_descriptor_size();
  std::vector<bool> tables_present(table_count, false);

  for (const auto& feature_config : config.feature_descriptor()) {
    const int table_id = feature_config.table_id();
    const auto& input_shape = feature_config.input_shape();
    if (table_id < 0 || table_id >= table_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid table_id: %d found in feature_descriptor: %s, all table_ids "
          "must be in the range[0, %d)",
          table_id, feature_config.ShortDebugString(), table_count));
    }
    if (input_shape.empty()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "The input_shape field cannot be empty in feature_descriptor: %s",
          feature_config.ShortDebugString()));
    }
    for (const int dim_size : input_shape) {
      if (dim_size <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "The input_shape dimension sizes must all be >0 in "
            "feature_descriptor: %s, found dimension size set to %d",
            feature_config.ShortDebugString(), dim_size));
      }
    }
    tables_present[table_id] = true;
  }  // feature_config

  for (int table_id = 0; table_id < table_count; ++table_id) {
    if (!tables_present[table_id]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No feature_descriptor fields found for table: %s (ID: %d) in "
          "the TPU embedding configuration.",
          config.table_descriptor(table_id).name(), table_id));
    }
  }
  return absl::OkStatus();
}

// Populates the feature_descriptor fields with default values when they have
// not been filled in by the user.
void PopulateFeatureDescriptors(tpu::TPUEmbeddingConfiguration* config) {
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    tpu::TPUEmbeddingConfiguration::FeatureDescriptor* feature_descriptor =
        config->add_feature_descriptor();
    feature_descriptor->set_table_id(table_id);
    feature_descriptor->add_input_shape(
        config->batch_size_per_tensor_core() *
        config->table_descriptor(table_id).num_features());
  }  // table_id
}

// Computes the input feature batch size based on the input feature shape. As
// we treat the last dimension as the reduction dimension, the batch size should
// be the product of all the axes except the last one.
std::vector<int> ComputeInputFeatureBatchSizes(
    const tpu::TPUEmbeddingConfiguration& config) {
  std::vector<int32_t> input_feature_batch_sizes;
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    const int32_t batch_size =
        absl::c_accumulate(config.feature_descriptor(i).input_shape(),
                           /*init=*/1, std::multiplies<>());
    input_feature_batch_sizes.push_back(batch_size);
  }
  return input_feature_batch_sizes;
}

// Computes the TensorCore batch size as the GCD of all input feature batch
// sizes.
int ComputeBatchSizePerTensorCore(
    absl::Span<const int> input_feature_batch_sizes) {
  uint32_t batch_size = input_feature_batch_sizes[0];
  for (const uint32_t input_feature_batch_size : input_feature_batch_sizes) {
    batch_size =
        tensorflow::MathUtil::GCD(batch_size, input_feature_batch_size);
  }
  return batch_size;
}

// Computes the TPU feature counts per user table as the sum of the TPU feature
// counts of the constituent input features. The TPU feature count for an input
// feature is the ratio of the batch size for that input feature to the batch
// size per TensorCore.
std::vector<int> ComputeTpuFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config,
    absl::Span<const int> input_feature_batch_sizes,
    int batch_size_per_tensor_core) {
  DCHECK_EQ(input_feature_batch_sizes.size(), config.feature_descriptor_size());
  std::vector<int> tpu_feature_counts(config.table_descriptor_size(), 0);
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    DCHECK_EQ(input_feature_batch_sizes[i] % batch_size_per_tensor_core, 0);
    tpu_feature_counts[config.feature_descriptor(i).table_id()] +=
        (input_feature_batch_sizes[i] / batch_size_per_tensor_core);
  }
  return tpu_feature_counts;
}

// Populates default values for batch_size_per_tensor_core and
// TableDescriptor.num_features when they have not been filled in by the user.
// The batch_size_per_tensor_core is computed as the GCD of the batch sizes of
// all input features.
void PopulateBatchSizeAndFeatureCounts(tpu::TPUEmbeddingConfiguration* config) {
  const std::vector<int> input_feature_batch_sizes =
      ComputeInputFeatureBatchSizes(*config);
  const int batch_size_per_tensor_core =
      ComputeBatchSizePerTensorCore(input_feature_batch_sizes);
  const std::vector<int> tpu_feature_counts = ComputeTpuFeatureCounts(
      *config, input_feature_batch_sizes, batch_size_per_tensor_core);
  config->set_batch_size_per_tensor_core(batch_size_per_tensor_core);
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    auto* table_config = config->mutable_table_descriptor(table_id);
    table_config->set_num_features(tpu_feature_counts[table_id]);
  }  // table_id
}

}  // namespace

absl::Status PopulateMissingFieldsInTPUEmbeddingConfig(
    tpu::TPUEmbeddingConfiguration* config) {
  if (config->feature_descriptor_size() == 0) {
    // If the feature_descriptor list is empty, validate that the batch size and
    // feature counts have been set properly. then, populate the
    // feature_descriptor with appropriate values.
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCounts(*config));
    PopulateFeatureDescriptors(config);
  } else {
    // If the feature_descriptor list is non-empty, validate that the batch size
    // and feature counts have NOT been populated. Also, validate that the
    // feature descriptors have been set properly. Then, populate the batch size
    // and feature counts with appropriate values.
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCountsAreEmpty(*config));
    TF_RETURN_IF_ERROR(ValidateFeatureDescriptors(*config));
    PopulateBatchSizeAndFeatureCounts(config);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
