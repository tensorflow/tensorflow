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
#ifndef TENSORFLOW_CORE_UTIL_EINSUM_OP_UTIL_H_
#define TENSORFLOW_CORE_UTIL_EINSUM_OP_UTIL_H_

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using Labels = absl::InlinedVector<int, 8UL>;
using OperandLabels = absl::InlinedVector<Labels, 2UL>;
using LabelCounts = absl::InlinedVector<int, 8UL>;
using OperandLabelCounts = absl::InlinedVector<LabelCounts, 2UL>;

// Dummy axis label used to denote an ellipsis in an input or output subscript.
constexpr int kEllipsisLabel = -1;

// Each dimension is categorized into exactly one of five types based on
// whether its corresponding label is present in the input and/or the output
// subscripts.
enum EinsumDimensionType {
  // Batch dimensions are those present in two inputs as well as the output.
  // They are part of the batch dimensions during Tensor contraction. Such
  // dimensions may be broadcasting dimensions (those mapping to ellipsis)
  // or explicit batch dimensions corresponding to named axis labels.
  kBroadcasting = 0,
  kBatch = 1,
  // Free dimensions are present in exactly one of the inputs, and also the
  // output. These are non-contracted axes in the Tensor contraction.
  kFree = 2,
  // Contract dimensions are present in two inputs, but not the output. These
  // dimensions are contracted in Tensor contraction.
  kContract = 3,
  // Reduce dimensions are present in exactly one input; and not in the output
  // and are summed over prior to Tensor contraction.
  kReduce = 4,
};

// Parses and validates an einsum equation in explicit form.
Status ValidateEinsumEquation(
    const string& equation, absl::InlinedVector<string, 2UL>* input_subscripts,
    string* output_subscript);

// Parses and validates the equation and the input shapes. Single character
// labels are integerized and we populate input and output label subscripts
// and corresponding counts. Also create the mapping from (named) labels to
// their EinsumDimensionType.
Status ParseEinsumEquation(const string& equation, OperandLabels* input_labels,
                           Labels* output_labels,
                           std::vector<EinsumDimensionType>* label_types,
                           OperandLabelCounts* input_label_counts,
                           LabelCounts* output_label_counts,
                           absl::InlinedVector<bool, 2UL>* input_has_ellipsis,
                           bool* output_has_ellipsis);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_EINSUM_OP_UTIL_H_
