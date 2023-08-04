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
#include "tensorflow/lite/kernels/internal/transpose_utils.h"

namespace tflite {
namespace transpose_utils {

bool IsTranspose2DApplicable(const TransposeParams& params,
                             const RuntimeShape& input_shape, int* dim0,
                             int* dim1) {
  const int dims_cnt = input_shape.DimensionsCount();

  if (dims_cnt == 2) {
    *dim0 = input_shape.Dims(0);
    *dim1 = input_shape.Dims(1);
    return true;
  }

  const int first_perm = params.perm[0];
  for (int i = 1; i < dims_cnt; ++i) {
    int rebased = params.perm[i] - first_perm;
    if (rebased < 0) {
      rebased += dims_cnt;
    }
    if (rebased != i) {
      return false;
    }
  }
  *dim0 = 1;
  *dim1 = 1;
  for (int i = 0; i < dims_cnt; ++i) {
    if (i < first_perm) {
      *dim0 *= input_shape.Dims(i);
    } else {
      *dim1 *= input_shape.Dims(i);
    }
  }
  return true;
}

void RemoveOneSizeDimensions(RuntimeShape* input_shape,
                             RuntimeShape* output_shape,
                             TransposeParams* params) {
  const int dims_cnt = input_shape->DimensionsCount();
  TFLITE_DCHECK_EQ(params->perm_count, dims_cnt);

  bool foundOneSizeDim = false;
  for (int i = 0; i < dims_cnt; ++i) {
    if (input_shape->Dims(i) == 1) {
      foundOneSizeDim = true;
      break;
    }
  }

  // Return here if there is no one size dimension.
  if (!foundOneSizeDim) return;

  // Handle the case where all the dimension size is one.
  if (input_shape->FlatSize() == 1) {
    input_shape->Resize(1);
    input_shape->SetDim(0, 1);
    output_shape->Resize(1);
    output_shape->SetDim(0, 1);
    params->perm_count = 1;
    params->perm[0] = 0;
    return;
  }

  // Resize input shape.
  int new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i) {
    if (input_shape->Dims(i) == 1) {
      continue;
    }
    input_shape->SetDim(new_dims_cnt, input_shape->Dims(i));
    ++new_dims_cnt;
  }
  input_shape->Resize(new_dims_cnt);

  // Resize output shape and re-calculate the perm parameter.
  TransposeParams new_params;
  new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i) {
    if (output_shape->Dims(i) == 1) {
      continue;
    }
    new_params.perm[new_dims_cnt] = params->perm[i];
    output_shape->SetDim(new_dims_cnt, output_shape->Dims(i));
    ++new_dims_cnt;
  }
  output_shape->Resize(new_dims_cnt);
  new_params.perm_count = new_dims_cnt;

  for (int i = 0; i < new_dims_cnt; ++i) {
    int min_val_idx = -1;
    for (int j = 0; j < new_dims_cnt; ++j) {
      if (new_params.perm[j] >= i &&
          (min_val_idx == -1 ||
           new_params.perm[min_val_idx] > new_params.perm[j])) {
        min_val_idx = j;
      }
    }
    new_params.perm[min_val_idx] = i;
  }
  *params = new_params;
}

size_t Flatten(const RuntimeShape& input_shape,
               const RuntimeShape& output_shape, const TransposeParams& params,
               RuntimeShape* non_flatten_input_shape,
               RuntimeShape* non_flatten_output_shape,
               TransposeParams* non_flatten_params) {
  // Calculate the total size of non-flatten dimensions.
  int skip_dims_cnt = 0;
  size_t flat_size = input_shape.FlatSize();
  for (int i = 0; i < params.perm_count; ++i) {
    if (params.perm[i] == i) {
      flat_size /= input_shape.Dims(i);
      ++skip_dims_cnt;
    } else {
      break;
    }
  }

  // Shrink the shapes and re-calculate the perm parameter.
  const int new_dims_cnt = params.perm_count - skip_dims_cnt;
  non_flatten_input_shape->Resize(new_dims_cnt);
  non_flatten_output_shape->Resize(new_dims_cnt);
  non_flatten_params->perm_count = new_dims_cnt;

  for (int i = skip_dims_cnt; i < params.perm_count; ++i) {
    non_flatten_input_shape->SetDim(i - skip_dims_cnt, input_shape.Dims(i));
    non_flatten_output_shape->SetDim(i - skip_dims_cnt, output_shape.Dims(i));
    non_flatten_params->perm[i - skip_dims_cnt] =
        params.perm[i] - skip_dims_cnt;
  }

  return flat_size;
}

}  // namespace transpose_utils

}  // namespace tflite
