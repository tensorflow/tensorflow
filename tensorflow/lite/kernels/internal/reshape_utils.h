/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_RESHAPE_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_RESHAPE_UTILS_H_

#include <cstddef>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace reshape_internal {

// Resolves a reshape output shape in place.
//
// This helper implements the reshape shape contract shared by Prepare() and
// Eval(): the requested output shape may contain at most one -1 dimension,
// every other dimension must be non-negative, zero dimensions are allowed, the
// inferred dimension must fit in the int shape representation, and input/output
// element counts must match without overflowing the size_t intermediates.
inline TfLiteStatus ResolveOutputShape(TfLiteContext* context,
                                       const RuntimeShape& input_shape,
                                       TfLiteIntArray& output_shape) {
  size_t num_input_elements = 0;
  TF_LITE_ENSURE_MSG(context, input_shape.CheckedFlatSize(num_input_elements),
                     "num_input_elements overflowed");
  CheckedInt<size_t> non_zero_num_input_elements = 1;
  for (int i = 0; i < input_shape.DimensionsCount(); ++i) {
    const int value = input_shape.Dims(i);
    if (value != 0) {
      non_zero_num_input_elements *= value;
    }
  }
  TF_LITE_ENSURE_MSG(context, !non_zero_num_input_elements.Overflow(),
                     "non_zero_num_input_elements overflowed");

  CheckedInt<size_t> non_zero_num_output_elements = 1;
  CheckedInt<size_t> num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < output_shape.size; ++i) {
    const int value = output_shape.data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
      continue;
    }
    TF_LITE_ENSURE_MSG(context, value >= 0,
                       "output shape contains negative dimension");
    if (value != 0) {
      non_zero_num_output_elements *= value;
    }
    num_output_elements *= value;
  }
  TF_LITE_ENSURE_MSG(context, !non_zero_num_output_elements.Overflow(),
                     "non_zero_num_output_elements overflowed");
  TF_LITE_ENSURE_MSG(context, !num_output_elements.Overflow(),
                     "num_output_elements overflowed");

  if (stretch_dim != -1) {
    if (num_input_elements == 0 && num_output_elements.Value() != 0) {
      output_shape.data[stretch_dim] = 0;
    } else {
      const size_t inferred_dim = non_zero_num_input_elements.Value() /
                                  non_zero_num_output_elements.Value();
      const CheckedInt<int> checked_inferred_dim(inferred_dim);
      TF_LITE_ENSURE_MSG(context, !checked_inferred_dim.Overflow(),
                         "inferred reshape dimension overflowed");
      output_shape.data[stretch_dim] = checked_inferred_dim.Value();
    }
    num_output_elements *= output_shape.data[stretch_dim];
    TF_LITE_ENSURE_MSG(context, !num_output_elements.Overflow(),
                       "num_output_elements overflowed");
  }

  TF_LITE_ENSURE_MSG(context, num_input_elements == num_output_elements.Value(),
                     "num_input_elements != num_output_elements (%zu != %zu)",
                     num_input_elements, num_output_elements.Value());
  return kTfLiteOk;
}

}  // namespace reshape_internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_RESHAPE_UTILS_H_
