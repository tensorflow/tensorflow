/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <numeric>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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

    const int64 num_rows = input.dimension(0);  // generally batch_size
    const int64 num_cols = input.dimension(1);

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

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    // Special case for k == 1.
    if (k == 1) {
#ifdef EIGEN_HAS_INDEX_LIST
      typename Eigen::IndexList<Eigen::type2index<1>> reduce_on_cols;
      typename Eigen::IndexList<int, Eigen::type2index<1>> rows_by_one;
      rows_by_one.set(0, num_rows);
#else
      Eigen::array<int, 1> reduce_on_cols = {1};
      Eigen::array<int, 1> rows_by_one = {static_cast<int>(num_rows), 1};
#endif

      values.device(d) =
          input.maximum(/*dims=*/reduce_on_cols).eval().reshape(rows_by_one);
      // Get the indices of the maximum values.
      for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
          if (values(r, 0) == input(r, c)) {
            indices(r, 0) = c;
            break;
          }
        }
      }

      return;
    }

    auto SortIndices = [&, context](int start_batch, int limit_batch) {
      for (int32 b = start_batch; b < limit_batch; ++b) {
        const T* input_data = &input(b, 0);
        const auto comp = [input_data](const int32 a, const int32 b) {
          return input_data[a] > input_data[b];
        };
        gtl::TopN<int32, decltype(comp)> filter(k, comp);
        // TODO(ebrevdo): For large k < num_cols, instead of using
        // TopN, it may be faster to create a temporary vector of
        // values 0..num_cols - 1 and then use std::partial_sort_copy
        // of this into indices. Choosing the appropriate minimum k or
        // ratio of k/num_cols will require some experimentation.
        if (k == num_cols) {
          // Set the initial array of indices 0 ... k - 1.
          std::iota(&indices(b, 0), &indices(b, k), 0);
          // Use an in-place sort.
          std::sort(&indices(b, 0), &indices(b, k), comp);
        } else {
          // Use the TopN heap object to sort.
          filter.reserve(num_cols);
          for (int32 c = 0; c < num_cols; ++c) {
            filter.push(c);
          }

          int32 i = 0;
          if (sorted_) {
            std::unique_ptr<std::vector<int32>> top_k(filter.Extract());
            for (auto top_k_it = top_k->begin(); top_k_it != top_k->end();
                 ++top_k_it, ++i) {
              indices(b, i) = *top_k_it;
            }
          } else {
            for (auto top_k_it = filter.unsorted_begin();
                 top_k_it != filter.unsorted_end(); ++top_k_it, ++i) {
              indices(b, i) = *top_k_it;
            }
          }
        }
        // Now that the indices are sorted, copy the values over in
        // sorted order.
        std::transform(&indices(b, 0), &indices(b, k), &values(b, 0),
                       [b, &input](const int32 loc) { return input(b, loc); });
      }  // for (int32 b = ...
    };

    // Guesstimate of cost; 4*N*log(K) where N == num_cols.
    // If K == N, assume the cost is N*log(K + 1).
    const int64 cmp_cost = 3 * Eigen::TensorOpCost::AddCost<int32>() +
                           Eigen::TensorOpCost::AddCost<T>();
    const int64 base_cost =
        cmp_cost *
        static_cast<int64>(num_cols *
                           Eigen::numext::log2(static_cast<float>(k + 1)));
    const int64 sort_cost = (k == num_cols) ? base_cost : 4 * base_cost;
    const int64 copy_cost = 2 * k * Eigen::TensorOpCost::AddCost<T>();
    const int64 total_cost = sort_cost + copy_cost;
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_rows,
          total_cost, SortIndices);
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
