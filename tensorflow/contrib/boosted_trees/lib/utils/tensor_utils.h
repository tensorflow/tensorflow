// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_TENSOR_UTILS_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_TENSOR_UTILS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

class TensorUtils {
 public:
  // Read an input list into a vector of tensors.
  static std::vector<Tensor> OpInputListToTensorVec(
      const OpInputList& input_list);

  // Reads the dense float features input list.
  static Status ReadDenseFloatFeatures(OpKernelContext* const context,
                                       OpInputList* features_list);

  // Reads the sparse float features input list.
  static Status ReadSparseFloatFeatures(OpKernelContext* const context,
                                        OpInputList* features_indices_list,
                                        OpInputList* feature_values_list,
                                        OpInputList* feature_shapes_list);

  // Reads the sparse int features input list.
  static Status ReadSparseIntFeatures(OpKernelContext* const context,
                                      OpInputList* features_indices_list,
                                      OpInputList* feature_values_list,
                                      OpInputList* feature_shapes_list);

  // Infers the batch size by looking at the op input features.
  static int64 InferBatchSize(
      const OpInputList& dense_float_features_list,
      const OpInputList& sparse_float_feature_shapes_list,
      const OpInputList& sparse_int_feature_shapes_list);
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_TENSOR_UTILS_H_
