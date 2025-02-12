/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/padding.h"

#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

absl::Status GetPaddingFromString(absl::string_view str_value, Padding* value) {
  if (str_value == "SAME") {
    *value = SAME;
  } else if (str_value == "VALID") {
    *value = VALID;
  } else if (str_value == "EXPLICIT") {
    *value = EXPLICIT;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding type");
  }
  return absl::OkStatus();
}

absl::Status CheckValidPadding(Padding padding_type,
                               const std::vector<int64_t>& explicit_paddings,
                               int num_dims, TensorFormat data_format) {
  if (padding_type == Padding::EXPLICIT) {
    const int num_paddings = explicit_paddings.size();
    if (num_paddings != 2 * num_dims) {
      return errors::InvalidArgument(
          "explicit_paddings attribute must contain ", 2 * num_dims,
          " values, but got: ", explicit_paddings.size());
    }
    for (int64_t padding_value : explicit_paddings) {
      if (padding_value < 0) {
        return errors::InvalidArgument(
            "All elements of explicit_paddings must be nonnegative");
      }
    }
    const int32_t batch_index = GetTensorBatchDimIndex(num_dims, data_format);
    const int32_t depth_index = GetTensorFeatureDimIndex(num_dims, data_format);
    if (explicit_paddings[2 * batch_index] != 0 ||
        explicit_paddings[2 * batch_index + 1] != 0 ||
        explicit_paddings[2 * depth_index] != 0 ||
        explicit_paddings[2 * depth_index + 1] != 0) {
      return errors::InvalidArgument(
          "Nonzero explicit padding in the batch or depth dimensions is not "
          "supported");
    }
  } else if (!explicit_paddings.empty()) {
    return errors::InvalidArgument(
        "explicit_paddings attribute must be empty if the padding attribute is "
        "not EXPLICIT");
  }
  return absl::OkStatus();
}

string GetPaddingAttrString() { return "padding: {'SAME', 'VALID'}"; }

string GetPaddingAttrStringWithExplicit() {
  return "padding: {'SAME', 'VALID', 'EXPLICIT'}";
}

string GetExplicitPaddingsAttrString() {
  return "explicit_paddings: list(int) = []";
}

}  // end namespace tensorflow
