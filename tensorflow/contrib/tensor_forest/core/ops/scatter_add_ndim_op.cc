// Copyright 2016 Google Inc. All Rights Reserved.
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
// ScatterAddNdim implements a scatter_add that can operate on sparse
// updates without being limited to the first dimension for indices.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

REGISTER_OP("ScatterAddNdim")
  .Input("input: Ref(float)")
  .Input("indices: int32")
  .Input("deltas: float")

  .Doc(R"doc(
  Add elements in deltas to mutable input according to indices.

  input: A N-dimensional float tensor to mutate.
  indices:= A 2-D int32 tensor. The size of dimension 0 is the number of
    deltas, the size of dimension 1 is the rank of the input.  `indices[i]`
    gives the coordinates of input that `deltas[i]` should add to
  deltas: `deltas[i]` is the value to add to input at index indices[i][:]
)doc");


class ScatterAddNdim : public OpKernel {
 public:
  explicit ScatterAddNdim(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor input_tensor = context->mutable_input(0, false);
    const Tensor& indices_tensor = context->input(1);
    const Tensor& deltas_tensor = context->input(2);

    OP_REQUIRES(context, deltas_tensor.shape().dims() == 1,
                errors::InvalidArgument(
                    "deltas should be one-dimensional"));
    if (indices_tensor.shape().dim_size(0) > 0) {
      OP_REQUIRES(context, indices_tensor.shape().dims() == 2,
                  errors::InvalidArgument(
                      "indices should be two-dimensional"));
      OP_REQUIRES(
          context,
          indices_tensor.shape().dim_size(1) == input_tensor.shape().dims(),
          errors::InvalidArgument(
              "Number of indices dimensions should be the same as input "
              "rank."));
      OP_REQUIRES(
          context,
          indices_tensor.shape().dim_size(0) ==
          deltas_tensor.shape().dim_size(0),
          errors::InvalidArgument(
              "Number of updates should be same as number of indices."));
    } else {
      return;
    }

    auto input = input_tensor.flat<float>();

    const auto indices = indices_tensor.tensor<int32, 2>();
    const auto deltas = deltas_tensor.unaligned_flat<float>();

    const int32 num_dims = indices_tensor.shape().dim_size(1);

    // Calculate index multipliers.
    std::vector<int32> multipliers;
    int32 last_size = input.size();

    for (int32 j = 0; j < num_dims; j++) {
      const int32 m = last_size / input_tensor.shape().dim_size(j);
      multipliers.push_back(m);
      last_size = m;
    }

    // Perform updates.
    for (int32 i = 0; i < indices_tensor.shape().dim_size(0); i++) {
      int32 index = 0;
      for (int32 j = 0; j < num_dims; j++) {
        index += indices(i, j) * multipliers[j];
      }
      input(index) += deltas(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ScatterAddNdim").Device(DEVICE_CPU),
                        ScatterAddNdim);
}  // namespace tensorflow
