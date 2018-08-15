/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_WINDOW_DATASET_H_
#define TENSORFLOW_CORE_KERNELS_DATA_WINDOW_DATASET_H_

#include <vector>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

// Creates a dataset representing an eagerly-collected window of elements.
//
// The `elements` argument defines the elements of the resulting
// dataset, which is stored in `out_dataset`.
//
// This dataset is constructed internally for use in datasets that
// build nested dataset expressions (e.g. the reducer function for
// GroupByWindowDataset). It efficiently supports multiple iterators on
// the same window without recomputation.
//
// REQUIRES: `output_types` must match the types of the respective
// element components in `elements`.
// REQUIRES: `output_shapes` must be compatible with the shapes of the
// respective element components in `elements`.a
Status NewWindowDataset(std::vector<std::vector<Tensor>> elements,
                        DataTypeVector output_types,
                        std::vector<PartialTensorShape> output_shapes,
                        DatasetBase** out_dataset);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_WINDOW_DATASET_H_
