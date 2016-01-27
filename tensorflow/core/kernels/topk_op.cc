/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/top_n.h"

namespace tensorflow {

template <typename T>
class TopK : public OpKernel {
 public:
  explicit TopK(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
    if (num_inputs() < 2) {  // k is an attr (TopK).
      OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    } else {  // k is an input (TopKV2), so we won't know it until Compute.
      k_ = -1;
    }
  }

  void Compute(OpKernelContext* context) override {
    int k = k_;
    if (num_inputs() >= 2) {
      const auto& k_in = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be scalar, got shape ",
                                          k_in.shape().DebugString()));
      k = k_in.scalar<int32>()();
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const auto& input_in = context->input(0);
    OP_REQUIRES(context, input_in.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_in.shape().DebugString()));
    OP_REQUIRES(context, input_in.dim_size(input_in.dims() - 1) >= k,
                errors::InvalidArgument("input must have at least k columns"));

    const auto& input = input_in.flat_inner_dims<T>();

    const auto num_rows = input.dimension(0);  // generally batch_size
    const auto num_cols = input.dimension(1);

    TensorShape output_shape = input_in.shape();
    output_shape.set_dim(input_in.dims() - 1, k);
    Tensor* values_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &values_out));
    Tensor* indices_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &indices_out));

    // Nothing to do for top-nothing.
    if (k == 0) return;

    auto values = values_out->flat_inner_dims<T>();
    auto indices = indices_out->flat_inner_dims<int32>();
    gtl::TopN<std::pair<T, int32>> filter(k);
    for (int r = 0; r < num_rows; r++) {
      for (int32 c = 0; c < num_cols; ++c) {
        // The second element is the negated index, so that lower-index elements
        // are considered larger than higher-index elements in case of ties.
        filter.push(std::make_pair(input(r, c), -c));
      }

      int32 i = 0;
      if (sorted_ && k > 1) {
        std::unique_ptr<std::vector<std::pair<T, int32>>> top_k(
            filter.Extract());
        for (auto top_k_it = top_k->begin(); top_k_it != top_k->end();
             ++top_k_it, ++i) {
          values(r, i) = top_k_it->first;
          indices(r, i) = -top_k_it->second;
        }
      } else {
        for (auto top_k_it = filter.unsorted_begin();
             top_k_it != filter.unsorted_end(); ++top_k_it, ++i) {
          values(r, i) = top_k_it->first;
          indices(r, i) = -top_k_it->second;
        }
      }
      filter.Reset();
    }
  }

 private:
  int k_;
  bool sorted_;
};

#define REGISTER_KERNELS_NAME(name, type) \
  REGISTER_KERNEL_BUILDER(                \
      Name(#name).Device(DEVICE_CPU).TypeConstraint<type>("T"), TopK<type>)

#define REGISTER_KERNELS(type)       \
  REGISTER_KERNELS_NAME(TopK, type); \
  REGISTER_KERNELS_NAME(TopKV2, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS_TO_NAME
#undef REGISTER_KERNELS

}  // namespace tensorflow
