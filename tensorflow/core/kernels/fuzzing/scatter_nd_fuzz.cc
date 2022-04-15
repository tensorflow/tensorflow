/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzScatterNd : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto indices =
        tensorflow::ops::Placeholder(scope.WithOpName("indices"), DT_INT32);
    auto updates =
        tensorflow::ops::Placeholder(scope.WithOpName("updates"), DT_INT32);
    auto shape =
        tensorflow::ops::Placeholder(scope.WithOpName("shape"), DT_INT32);
    (void)tensorflow::ops::ScatterNd(scope.WithOpName("output"), indices,
                                     updates, shape);
  }

  void FuzzImpl(const uint8_t* data, size_t size) override {
    // This op's runtime is heavily determined by the shape of the tensor
    // arguments and almost not at all by the values of those tensors. Hence,
    // the fuzzing data here is only used to determine the shape of the
    // arguments and the output and the data of these tensors is just a constant
    // value. Furthermore, the shape of the updates_tensor tensor is fully
    // determined by the contents of the shape_tensor and the shape of the
    // indices_tensor. Rather than using random values for the
    // updates_tensor.shape and getting most of the fuzz runs stopped in the
    // check, it's better to just create a proper update_tensor.
    if (size < 1) {
      return;
    }

    // First element of the data buffer gives the number of dimensions of the
    // shape tensor.
    size_t i;
    size_t data_ix = 0;
    size_t shape_dims = 1 + (data[data_ix++] % kMaxShapeDims);
    Tensor shape_tensor(tensorflow::DT_INT32,
                        TensorShape({static_cast<int64_t>(shape_dims)}));

    // Check that we have enough elements left for the shape tensor
    if (data_ix + shape_dims >= size) {
      return;  // not enough elements, no fuzz
    }

    // Subsequent elements give the contents of the shape tensor.
    // To not get out of memory, reduce all dimensions to at most kMaxDim
    auto flat_shape = shape_tensor.flat<int32>();
    for (i = 0; i < shape_dims; i++) {
      flat_shape(i) = data[data_ix++] % kMaxDim;
    }

    // Next, we have to fill in the indices tensor. Take the next element from
    // the buffer to represent the rank of this tensor.
    if (data_ix >= size) {
      return;
    }
    size_t indices_rank = 1 + (data[data_ix++] % kMaxIndicesRank);

    // Now, read the dimensions of the indices_tensor
    if (data_ix + indices_rank >= size) {
      return;
    }
    std::vector<int64_t> indices_dims;
    size_t num_indices = 1;
    for (i = 0; i < indices_rank; i++) {
      // Modulo kMaxDim to not request too much memory
      int64_t dim = data[data_ix++] % kMaxDim;
      num_indices *= dim;
      indices_dims.push_back(dim);
    }
    Tensor indices_tensor(tensorflow::DT_INT32, TensorShape(indices_dims));

    // Rest of the buffer is used to fill in the indices_tensor
    auto flat_indices = indices_tensor.flat<int32>();
    for (i = 0; i < num_indices && data_ix < size; i++) {
      flat_indices(i) = data[data_ix++];
    }
    for (; i < num_indices; i++) {
      flat_indices(i) = 0;  // ensure that indices_tensor has all values
    }

    // Given the values in the shape_tensor and the dimensions of the
    // indices_tensor, the shape of updates_tensor is fixed.
    num_indices = 1;
    std::vector<int64_t> updates_dims;
    for (i = 0; i < indices_rank - 1; i++) {
      updates_dims.push_back(indices_dims[i]);
      num_indices *= indices_dims[i];
    }
    int64_t last = indices_dims[indices_rank - 1];
    for (i = last; i < shape_dims; i++) {
      updates_dims.push_back(flat_shape(i));
      num_indices *= flat_shape(i);
    }
    Tensor updates_tensor(tensorflow::DT_INT32, TensorShape(updates_dims));

    // We don't care about the values in the updates_tensor, make them all be 1
    auto flat_updates = updates_tensor.flat<int32>();
    for (i = 0; i < num_indices; i++) {
      flat_updates(i) = 1;
    }

    RunInputs({{"indices", indices_tensor},
               {"updates", updates_tensor},
               {"shape", shape_tensor}});
  }

 private:
  const size_t kMaxShapeDims = 5;
  const size_t kMaxIndicesRank = 3;
  const size_t kMaxDim = 10;
};

STANDARD_TF_FUZZ_FUNCTION(FuzzScatterNd);

}  // end namespace fuzzing
}  // end namespace tensorflow
