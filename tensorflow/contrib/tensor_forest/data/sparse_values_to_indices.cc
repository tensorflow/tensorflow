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
// Converts strings of arbitrary length to float values by
// hashing and cramming bits.
#include <functional>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using tensorforest::CheckTensorBounds;
using tensorforest::Initialize;

int64 Convert(const string& in, int32 offset_bits) {
  std::size_t hashed = std::hash<string>()(in);
  // Mask off the offset_bits msb's
  int64 mask = static_cast<int64>(pow(2, offset_bits) - 1)
               << (32 - offset_bits);
  // TODO(gilberth): Use int64 to store feature indices in tensor_forest.
  // Only use the lower 31 bits because that's what we currently store as
  // feature indices.
  mask = ~mask & 0x7FFFFFFF;
  return static_cast<int64>(hashed & mask);
}

void Evaluate(const Tensor& sparse_indices, const Tensor& sparse_values,
              Tensor output_data, int64 offset, int32 offset_bits, int32 start,
              int32 end) {
  auto out_data = output_data.tensor<int64, 2>();
  const auto indices = sparse_indices.tensor<int64, 2>();
  const auto values = sparse_values.unaligned_flat<string>();

  for (int32 i = start; i < end; ++i) {
    out_data(i, 0) = indices(i, 0);
    out_data(i, 1) = Convert(values(i), offset_bits) + offset;
  }
}

REGISTER_OP("SparseValuesToIndices")
    .Attr("offset_bits: int")
    .Input("sparse_indices: int64")
    .Input("sparse_values: string")
    .Input("offset: int64")
    .Output("output_indices: int64")
    .Output("output_values: float")

    .Doc(R"doc(
   Converts string values to sparse indices in a bit vector.

   offset_bits: The number of bits being used for offsets, which tells us
     how many bits we can use to represent the string values.
   sparse_indices: The original sparse indices (2-d tensor).
   sparse_values: A batch of string values as a 1-d tensor.
   offset: An offset value to apply to the feature index.
   output_indices: A tensor of the same shape as sparse_indices where
     output_indices[i][0] is the same as sparse_indices and output_indices[i][1]
     is the integer value of the corresponding sparse_values.

)doc");

class SparseValuesToIndices : public OpKernel {
 public:
  explicit SparseValuesToIndices(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("offset_bits", &offset_bits_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& sparse_indices = context->input(0);
    const Tensor& sparse_values = context->input(1);
    const Tensor& offset_tensor = context->input(2);

    // Check inputs.
    OP_REQUIRES(
        context, sparse_indices.shape().dims() == 2,
        errors::InvalidArgument("sparse_indices should be two-dimensional"));
    OP_REQUIRES(
        context, sparse_values.shape().dims() == 1,
        errors::InvalidArgument("sparse_values should be one-dimensional"));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, sparse_indices)) return;
    if (!CheckTensorBounds(context, sparse_values)) return;
    if (!CheckTensorBounds(context, offset_tensor)) return;

    Tensor* output_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, sparse_indices.shape(),
                                                     &output_indices));
    Tensor* output_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, sparse_values.shape(),
                                                     &output_values));
    // There doesn't seem to be a great way to get a tensor of 1's with the same
    // shape as a string tensor with unknown dimensions, so we just do it here
    // in this op.
    Initialize<float>(*output_values, 1.0);

    const int64 offset = offset_tensor.unaligned_flat<int64>()(0);

    // Evaluate input data in parallel.
    const int32 num_data =
        static_cast<int32>(sparse_values.shape().dim_size(0));
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    if (num_threads <= 1) {
      Evaluate(sparse_indices, sparse_values, *output_indices, offset, 0,
               offset_bits_, num_data);
    } else {
      auto work = [this, &sparse_indices, sparse_values, output_indices, offset,
                   num_data](int64 start, int64 end) {
        CHECK(start <= end);
        CHECK(end <= num_data);
        Evaluate(sparse_indices, sparse_values, *output_indices, offset,
                 offset_bits_, static_cast<int32>(start),
                 static_cast<int32>(end));
      };
      Shard(num_threads, worker_threads->workers, num_data, 100, work);
    }
  }

 private:
  int32 offset_bits_;
};

REGISTER_KERNEL_BUILDER(Name("SparseValuesToIndices").Device(DEVICE_CPU),
                        SparseValuesToIndices);

}  // namespace tensorflow
