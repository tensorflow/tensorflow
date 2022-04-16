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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_WHERE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_WHERE_H_

template <typename D, typename T>
void SelectTrueCoords(const RuntimeShape& input_condition_shape,
                      const D* input_condition_data, T* output_data) {
  const size_t size = input_condition_shape.FlatSize();
  if (size == 0) {
    // Dimension is zero, in which case we don't need to output.
    return;
  }
  const size_t cond_rank = input_condition_shape.DimensionsCount();

  std::vector<int> dims_to_count(cond_rank, 0);
  int cur_flat_size = size;
  for (int i = 0; i < cond_rank; ++i) {
    dims_to_count[i] = cur_flat_size / input_condition_shape.Dims(i);
    cur_flat_size = dims_to_count[i];
  }

  int output_index = 0;
  for (int i = 0; i < size; ++i) {
    if (input_condition_data[i] != D(0)) {
      // Insert the coordinate of the current item (row major) into output.
      int flat_index = i;
      for (int j = 0; j < cond_rank; ++j) {
        int coord_j = flat_index / dims_to_count[j];
        output_data[output_index * cond_rank + j] = coord_j;
        flat_index %= dims_to_count[j];
      }
      output_index++;
    }
  }
}

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_WHERE_H_
