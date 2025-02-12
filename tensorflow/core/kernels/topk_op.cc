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

#include "tensorflow/core/kernels/topk_op.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tidx>
class TopK : public OpKernel {
 public:
  explicit TopK(OpKernelConstruction* context) : OpKernel(context) {
    // Allow "sorted" to be an optional attribute for use with ApproxTopK
    // which has no such attribute.
    auto status = context->GetAttr("sorted", &sorted_);
    if (!status.ok()) {
      sorted_ = true;  // Default to sorted, as required by ApproxTopK.
    }

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
      switch (k_in.dtype()) {
        case DT_INT16:
          k = k_in.scalar<int16_t>()();
          break;
        case DT_INT32:
          k = k_in.scalar<int32_t>()();
          break;
        case DT_INT64:
          k = k_in.scalar<int64_t>()();
          break;
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument(
                          "k must have dtype in {int16, int32, int64}, got  ",
                          k_in.dtype()));
      }
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const auto& input_in = context->input(0);
    OP_REQUIRES(context, input_in.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_in.shape().DebugString()));
    OP_REQUIRES(context, input_in.dim_size(input_in.dims() - 1) >= k,
                errors::InvalidArgument(
                    "input must have at least k columns. Had ",
                    input_in.dim_size(input_in.dims() - 1), ", needed ", k));

    const auto& input = input_in.flat_inner_dims<T>();

    const int64_t num_rows = input.dimension(0);  // generally batch_size
    const int64_t num_cols = input.dimension(1);
    OP_REQUIRES(context, num_rows <= std::numeric_limits<Tidx>::max(),
                errors::InvalidArgument(
                    "First dimension of flattened input must be <= ",
                    std::numeric_limits<Tidx>::max(), ", got ", num_rows));
    OP_REQUIRES(context, num_cols <= std::numeric_limits<Tidx>::max(),
                errors::InvalidArgument(
                    "Second dimension of flattened input must be <= ",
                    std::numeric_limits<Tidx>::max(), ", got ", num_cols));

    TensorShape output_shape = input_in.shape();
    output_shape.set_dim(input_in.dims() - 1, k);
    Tensor* values_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &values_out));
    Tensor* indices_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &indices_out));

    // Nothing to do for top-nothing or over nothing.
    if (k == 0 || num_rows == 0) return;

    auto values = values_out->flat_inner_dims<T>();
    auto indices = indices_out->flat_inner_dims<Tidx>();
    absl::Status s = functor::TopKFunctor<Device, T, Tidx>::Compute(
        context, sorted_, k, input, num_rows, num_cols, values, indices);
    OP_REQUIRES_OK(context, s);
  }

 private:
  int k_;
  bool sorted_;
};

namespace functor {

template <typename T, typename Tidx>
struct TopKFunctor<CPUDevice, T, Tidx> {
  static EIGEN_ALWAYS_INLINE absl::Status Compute(
      OpKernelContext* context, bool sorted, int k,
      const typename TTypes<T, 2>::ConstTensor& input, const int64_t num_rows,
      const int64_t num_cols, typename TTypes<T, 2>::Tensor values,
      typename TTypes<Tidx, 2>::Tensor indices) {
    const CPUDevice& d = context->eigen_device<CPUDevice>();

    // Special case for k == 1.
    if (k == 1) {
      typename Eigen::IndexList<Eigen::type2index<1>> reduce_on_cols;
      typename Eigen::IndexList<int, Eigen::type2index<1>> rows_by_one;
      rows_by_one.set(0, num_rows);

      values.device(d) =
          input.maximum(/*dims=*/reduce_on_cols).eval().reshape(rows_by_one);
      // Get the indices of the maximum values.
      for (int r = 0; r < num_rows; ++r) {
        indices(r, 0) = Tidx(0);
        for (int c = 0; c < num_cols; ++c) {
          if (values(r, 0) == input(r, c)) {
            indices(r, 0) = static_cast<Tidx>(c);
            break;
          }
        }
        values(r, 0) = input(r, indices(r, 0));
      }

      return absl::OkStatus();
    }

    auto SortIndices = [&](int64_t start_batch, int64_t limit_batch) {
      for (int32_t b = start_batch; b < limit_batch; ++b) {
        const T* input_data = &input(b, 0);
        const auto stable_comp = [input_data](const int32_t a,
                                              const int32_t b) {
          if (input_data[b] < input_data[a]) {
            return true;
          } else if (input_data[b] > input_data[a]) {
            return false;
          } else {
            return a < b;
          }
        };
        const auto comp = [input_data](const int32_t a, const int32_t b) {
          return input_data[b] < input_data[a];
        };
        // TODO(ebrevdo): For large k < num_cols, instead of using
        // TopN, it may be faster to create a temporary vector of
        // values 0..num_cols - 1 and then use std::partial_sort_copy
        // of this into indices. Choosing the appropriate minimum k or
        // ratio of k/num_cols will require some experimentation.
        if (k == num_cols) {
          auto* begin = &indices(b, 0);
          auto* end = &indices(b, k);
          // Set the initial array of indices 0 ... k - 1.
          std::iota(begin, end, 0);
          // We want an in-place sort, but we can cheat because we're sorting
          // indices that started out sorted.  First, do a std::sort, which
          // is notably faster than std::stable_sort.
          std::sort(begin, end, comp);
          // Then, for runs of adjacent elements that were equal, sort the
          // indices in those runs in increasing order.
          for (auto* run_begin = begin; run_begin != end;) {
            auto* run_end = run_begin + 1;
            if (run_end == end) break;
            if (input_data[*run_begin] == input_data[*run_end]) {
              while (++run_end != end) {
                if (input_data[*run_begin] != input_data[*run_end]) break;
              }
              std::sort(run_begin, run_end);
            }
            run_begin = run_end;
          }
        } else {
          // Use the TopN heap object to sort.
          gtl::TopN<Tidx, decltype(stable_comp)> filter(k, stable_comp);
          filter.reserve(num_cols);
          for (Tidx c = 0; c < num_cols; ++c) {
            filter.push(c);
          }

          int32_t i = 0;
          if (sorted) {
            std::unique_ptr<std::vector<Tidx>> top_k(filter.Extract());
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
                       [b, &input](const Tidx loc) { return input(b, loc); });
      }  // for (Tidx b = ...
    };

    // Guesstimate of cost; 4*N*log(K) where N == num_cols.
    // If K == N, assume the cost is N*log(K + 1).
    const double cmp_cost = 3 * Eigen::TensorOpCost::AddCost<Tidx>() +
                            Eigen::TensorOpCost::AddCost<T>();
    const double base_cost =
        cmp_cost *
        static_cast<double>(num_cols *
                            Eigen::numext::log2(static_cast<float>(k + 1)));
    const double sort_cost = (k == num_cols) ? base_cost : 4 * base_cost;
    const double copy_cost = 2 * k * Eigen::TensorOpCost::AddCost<T>();
    const double total_cost = sort_cost + copy_cost;
    const int64_t final_cost = (total_cost >= static_cast<double>(kint64max))
                                   ? kint64max
                                   : static_cast<int64_t>(total_cost);
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_rows,
          final_cost, SortIndices);

    return absl::OkStatus();
  }
};

}  // namespace functor

#define REGISTER_KERNELS_NAME(name, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name(#name)                                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("index_type"), \
                          TopK<CPUDevice, type, index_type>)

#define REGISTER_KERNELS_WITH_INDEX(type, index_type) \
  REGISTER_KERNELS_NAME(TopK, type, index_type);      \
  REGISTER_KERNELS_NAME(TopKV2, type, index_type);

#define REGISTER_APPROX_TOPK_KERNELS(type)                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("ApproxTopK")                                      \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<type>("T")                          \
          .AttrConstraint<int64_t>("reduction_dimension", -1) \
          .AttrConstraint<bool>("is_max_k", true),            \
      TopK<CPUDevice, type, int32>);

#define REGISTER_KERNELS(type)                \
  REGISTER_KERNELS_WITH_INDEX(type, int16);   \
  REGISTER_KERNELS_WITH_INDEX(type, int32);   \
  REGISTER_KERNELS_WITH_INDEX(type, int64_t); \
  REGISTER_APPROX_TOPK_KERNELS(type);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);

#undef REGISTER_TOPK_KERNELS_NAME
#undef REGISTER_TOPK_KERNELS_WITH_INDEX
#undef REGISTER_APPROX_TOPK_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
#define DECLARE_GPU_SPEC_WITH_INDEX(T, Tidx)                                   \
  template <>                                                                  \
  Status TopKFunctor<GPUDevice, T, Tidx>::Compute(                             \
      OpKernelContext* context, bool sorted, int k,                            \
      const typename TTypes<T, 2>::ConstTensor& input, const int64_t num_rows, \
      const int64_t num_cols, typename TTypes<T, 2>::Tensor values,            \
      typename TTypes<Tidx, 2>::Tensor indices);                               \
  extern template struct functor::TopKFunctor<GPUDevice, T, Tidx>;

#define DECLARE_GPU_SPEC(T) DECLARE_GPU_SPEC_WITH_INDEX(T, int32)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_INTEGRAL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_WITH_INDEX

}  // namespace functor

#define REGISTER_TOPK_KERNELS_WITH_INDEX(type, index_type)               \
  REGISTER_KERNEL_BUILDER(Name("TopK")                                   \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("index_type"), \
                          TopK<GPUDevice, type, index_type>);            \
  REGISTER_KERNEL_BUILDER(Name("TopKV2")                                 \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("index_type")  \
                              .HostMemory("k"),                          \
                          TopK<GPUDevice, type, index_type>)

#define REGISTER_APPROX_TOPK_KERNELS(type)                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("ApproxTopK")                                      \
          .Device(DEVICE_GPU)                                 \
          .TypeConstraint<type>("T")                          \
          .AttrConstraint<int64_t>("reduction_dimension", -1) \
          .AttrConstraint<bool>("is_max_k", true),            \
      TopK<GPUDevice, type, int32>);

#define REGISTER_TOPK_KERNELS(type) \
  REGISTER_TOPK_KERNELS_WITH_INDEX(type, int32)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TOPK_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_TOPK_KERNELS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_APPROX_TOPK_KERNELS);

#undef REGISTER_TOPK_KERNELS
#undef REGISTER_TOPK_KERNELS_WITH_INDEX
#undef REGISTER_APPROX_TOPK_KERNELS

#endif  // end GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
