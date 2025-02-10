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
#include "tensorflow/core/data/service/validate_utils.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::protobuf::util::MessageDifferencer;

absl::StatusOr<StructuredValue> DecodeElementSpec(
    const std::string& dataset_id, const std::string& encoded_spec) {
  if (encoded_spec.empty()) {
    return StructuredValue();
  }

  StructuredValue decoded_spec;
  if (!decoded_spec.ParsePartialFromString(encoded_spec)) {
    return errors::InvalidArgument("Failed to parse element_spec for dataset ",
                                   dataset_id, ": ", encoded_spec, ".");
  }
  return decoded_spec;
}

absl::Status ValidateElementSpec(const std::string& dataset_id,
                                 const std::string& encoded_spec1,
                                 const std::string& encoded_spec2) {
  if (encoded_spec1.empty() && encoded_spec2.empty()) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(StructuredValue element_spec1,
                      DecodeElementSpec(dataset_id, encoded_spec1));
  TF_ASSIGN_OR_RETURN(StructuredValue element_spec2,
                      DecodeElementSpec(dataset_id, encoded_spec2));

  MessageDifferencer differ;
  std::string diff;
  differ.ReportDifferencesToString(&diff);
  differ.set_message_field_comparison(MessageDifferencer::EQUIVALENT);
  differ.set_repeated_field_comparison(MessageDifferencer::AS_SET);
  differ.set_float_comparison(MessageDifferencer::APPROXIMATE);
  if (!differ.Compare(element_spec1, element_spec2)) {
    return errors::InvalidArgument(
        "Datasets with the same ID should have the same structure, got diff ",
        "for dataset ID ", dataset_id, " with different element_spec: ", diff,
        ". To fix this error, make sure you're registering the same dataset ",
        "with the same ID.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDatasetMetadata(const std::string& dataset_id,
                                     const DataServiceMetadata& metadata1,
                                     const DataServiceMetadata& metadata2) {
  TF_RETURN_IF_ERROR(ValidateElementSpec(dataset_id, metadata1.element_spec(),
                                         metadata2.element_spec()));
  MessageDifferencer differ;
  std::string diff;
  differ.ReportDifferencesToString(&diff);
  differ.IgnoreField(
      DataServiceMetadata::descriptor()->FindFieldByName("element_spec"));
  differ.set_message_field_comparison(MessageDifferencer::EQUIVALENT);
  differ.set_repeated_field_comparison(MessageDifferencer::AS_SET);
  differ.set_float_comparison(MessageDifferencer::APPROXIMATE);
  if (!differ.Compare(metadata1, metadata2)) {
    return errors::InvalidArgument(
        "Datasets with the same ID should have the same structure, got diff ",
        "for dataset ID ", dataset_id, ": ", diff, ". To fix this error, make ",
        "sure you're registering the same dataset with the same ID.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateMatchingDataset(const std::string& dataset_id,
                                     const DataServiceMetadata& metadata1,
                                     const DataServiceMetadata& metadata2) {
  return ValidateDatasetMetadata(dataset_id, metadata1, metadata2);
}

}  // namespace data
}  // namespace tensorflow
