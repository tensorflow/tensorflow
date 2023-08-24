/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_KERNELS_TEST_UTIL_H_

#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/util.h"

namespace tflite {

class ListOpModel : public MultiOpModel {
 public:
  // Populate tensor at given index as a TensorList.
  void PopulateListTensor(int index, absl::Span<const int> element_shape_data,
                          int num_elements, TfLiteType element_type);

  // Set a list element with given data.
  void ListSetItem(int index, int list_index, absl::Span<const int> item_dims,
                   TfLiteType item_type, const void* item_data);
};

// Gets the number of bytes required for a single element of `TfLiteType`,
// or `nullopt` if type does not have a fixed size.
std::optional<size_t> TfLiteTypeSizeOf(TfLiteType type);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_KERNELS_TEST_UTIL_H_
