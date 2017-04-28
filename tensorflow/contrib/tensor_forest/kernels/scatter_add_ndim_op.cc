// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;
using tensorforest::CheckTensorBounds;


class ScatterAddNdim : public OpKernel {
 public:
  explicit ScatterAddNdim(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor input_tensor = context->mutable_input(0, false);
    const Tensor& indices_tensor = context->input(1);
    const Tensor& deltas_tensor = context->input(2);

    if (indices_tensor.shape().dim_size(0) > 0) {
      OP_REQUIRES(context, indices_tensor.shape().dims() == 2,
                  errors::InvalidArgument(
                      "indices should be two-dimensional"));
      const int32 delta_dims = deltas_tensor.shape().dims();
      OP_REQUIRES(
          context,
          indices_tensor.shape().dim_size(1) + delta_dims ==
          input_tensor.shape().dims() + 1,
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

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_tensor)) return;
    if (!CheckTensorBounds(context, indices_tensor)) return;
    if (!CheckTensorBounds(context, deltas_tensor)) return;

    auto input = input_tensor.flat<float>();

    const auto indices = indices_tensor.tensor<int32, 2>();
    const auto deltas = deltas_tensor.unaligned_flat<float>();

    const int32 num_dims = static_cast<int32>(
        indices_tensor.shape().dim_size(1));

    // Figure out if indices don't specify a complete position in the
    // input tensor.
    int32 num_data_per_index = 1;
    for (int32 i = 0; i < input_tensor.shape().dims() - num_dims; ++i) {
      num_data_per_index *= input_tensor.shape().dim_size(num_dims + i);
    }

    // Calculate index multipliers.
    std::vector<int32> multipliers;
    OP_REQUIRES(
        context, input.size() < std::numeric_limits<int32>::max(),
        errors::InvalidArgument(
            "Input must contain less than 2^31 total elements"));
    int32 last_size = static_cast<int32>(input.size());

    for (int32 j = 0; j < num_dims; j++) {
      const int32 m = last_size / input_tensor.shape().dim_size(j);
      multipliers.push_back(m);
      last_size = m;
    }

    // Perform updates.
    for (int32 i = 0; i < indices_tensor.shape().dim_size(0); i++) {
      int32 start_index = 0;
      for (int32 j = 0; j < num_dims; j++) {
        start_index += indices(i, j) * multipliers[j];
      }
      for (int32 offset = 0; offset < num_data_per_index; ++offset) {
        const int32 input_index = start_index + offset;
        const int32 delta_index = i * num_data_per_index + offset;
        CHECK(input_index < input.size());
        CHECK(delta_index < deltas.size());
        input(input_index) += deltas(delta_index);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ScatterAddNdim").Device(DEVICE_CPU),
                        ScatterAddNdim);
}  // namespace tensorflow
