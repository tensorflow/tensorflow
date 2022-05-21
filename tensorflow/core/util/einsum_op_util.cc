/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/einsum_op_util.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

Status ValidateEinsumEquation(const string& equation,
                              gtl::InlinedVector<string, 2>* input_subscripts,
                              string* output_subscript) {
  gtl::InlinedVector<string, 2> inputs_and_output_subscripts =
      absl::StrSplit(equation, "->");
  if (inputs_and_output_subscripts.size() != 2) {
    return errors::InvalidArgument(
        "Expecting exactly one '->' in einsum equation: ", equation);
  }
  *output_subscript = std::move(inputs_and_output_subscripts[1]);
  *input_subscripts =
      absl::StrSplit(std::move(inputs_and_output_subscripts[0]), ',');
  if (input_subscripts->size() != 1 && input_subscripts->size() != 2) {
    return errors::InvalidArgument(
        "Expecting 1 or 2 input subscripts in equation '", equation,
        "' but got: ", input_subscripts->size());
  }
  return Status::OK();
}

// Returns the EinsumDimensionType given whether the corresponding label is
// present in exactly one input subscript (is_unique) and whether it is absent
// from the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
EinsumDimensionType GetDimensionType(bool is_removed, bool is_unique) {
  if (!is_removed && !is_unique)
    return kBatch;
  else if (!is_removed && is_unique)
    return kFree;
  else if (is_removed && !is_unique)
    return kContract;
  else  // is_removed && is_unique
    return kReduce;
}

// Maps the character labels to consecutive integers.
void MapToLabels(const string& subscript, Labels* labels,
                 absl::flat_hash_map<char, int>* label_mapping) {
  for (int i = 0; i < subscript.size(); ++i) {
    const char label_char = subscript[i];
    if (label_char == '.') {
      labels->push_back(kEllipsisLabel);
      i += 2;  // Skip next 2 characters as well.
      continue;
    }
    if (!label_mapping->contains(label_char)) {
      const int next_label = label_mapping->size();
      (*label_mapping)[label_char] = next_label;
    }
    const int mapped_label = (*label_mapping)[label_char];
    labels->push_back(mapped_label);
  }
}

Status ParseEinsumEquation(const string& equation, OperandLabels* input_labels,
                           Labels* output_labels,
                           std::vector<EinsumDimensionType>* label_types,
                           OperandLabelCounts* input_label_counts,
                           LabelCounts* output_label_counts,
                           gtl::InlinedVector<bool, 2>* input_has_ellipsis,
                           bool* output_has_ellipsis) {
  gtl::InlinedVector<string, 2> input_str;
  string output_str;
  TF_RETURN_IF_ERROR(ValidateEinsumEquation(equation, &input_str, &output_str));

  // Temporary map from single character labels to (consecutive) integer labels.
  absl::flat_hash_map<char, int> label_mapping;
  int num_inputs = input_str.size();
  input_labels->resize(num_inputs);

  // Map from single characters to integer labels.
  for (int i = 0; i < num_inputs; ++i) {
    MapToLabels(input_str[i], &input_labels->at(i), &label_mapping);
  }
  MapToLabels(output_str, output_labels, &label_mapping);

  // Compute counts for input and output labels.
  int num_labels = label_mapping.size();
  input_label_counts->resize(num_inputs);
  input_has_ellipsis->resize(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_label_counts->at(i).resize(num_labels);
    input_has_ellipsis->at(i) = false;
    for (const int label : input_labels->at(i)) {
      if (label != kEllipsisLabel)
        input_label_counts->at(i)[label] += 1;
      else
        input_has_ellipsis->at(i) = true;
    }
  }
  output_label_counts->resize(num_labels);
  *output_has_ellipsis = false;
  for (const int label : *output_labels) {
    if (label != kEllipsisLabel)
      output_label_counts->at(label) += 1;
    else
      *output_has_ellipsis = true;
  }

  // Map each label to a unique EinsumDimensionType.
  label_types->resize(num_labels);
  for (int label = 0; label < num_labels; ++label) {
    if (label == kEllipsisLabel) continue;
    bool removed = (*output_label_counts)[label] == 0;
    bool unique = num_inputs == 1 || (*input_label_counts)[0][label] == 0 ||
                  (*input_label_counts)[1][label] == 0;
    (*label_types)[label] = GetDimensionType(removed, unique);
  }
  return Status::OK();
}

}  // namespace tensorflow
