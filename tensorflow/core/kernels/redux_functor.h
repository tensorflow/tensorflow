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

#ifndef TENSORFLOW_CORE_KERNELS_REDUX_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_REDUX_FUNCTOR_H_

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace functor {

// Compute reduction over all outer dimensions.
// Example:
//   input: [32, 32, 256]
//   ->
//   output: [256]
template <typename T, typename AccumT, typename BinaryFunctor>
struct ReduceOuterDimensions {
  template <int num_dims>
  void operator()(const CPUDevice& device,
                  const Eigen::DSizes<Eigen::Index, num_dims>& input_dims,
                  const Tensor& input, Tensor* output) const {
    static_assert(num_dims >= 2, "Input dimensions must at least 2");

    // Compute inner and outer dim after reshaping into 2d tensor.
    int64 inner_dim = input_dims[num_dims - 1];
    int64 outer_dim = 1;
    for (int i = 0; i < num_dims - 1; ++i) outer_dim *= input_dims[i];

    // Compute block size along the outer dimension for efficiency.
    const int64 parallel_cell_size = inner_dim;
    const int64 total_workload = outer_dim * inner_dim;
    const int64 max_parallelism = total_workload / parallel_cell_size;

    const int64 min_block_workload = 2000;
    const int64 min_block_size =
        Eigen::divup(min_block_workload, parallel_cell_size);
    const int64 max_num_blocks =
        std::min(max_parallelism, Eigen::divup(total_workload, min_block_size));

    // Do not create more blocks than there are threads in a pool.
    const int64 num_threads = device.numThreads();
    const int64 num_blocks = std::min(max_num_blocks, num_threads);

    // Block size along the outer dimension.
    const int64 outer_block_size = Eigen::divup(outer_dim, num_blocks);

    const T* input_data = input.template flat<T>().data();

    // Allocate temporary buffer for partial reductions.
    Tensor buffer(DataTypeToEnum<AccumT>::v(), {num_blocks, inner_dim});
    buffer.template flat<AccumT>().setZero();
    AccumT* buffer_data = buffer.template flat<AccumT>().data();

    using Buffer = Eigen::TensorMap<
        Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>,
        Eigen::Unaligned>;

    using Input = Eigen::TensorMap<
        Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::Index>,
        Eigen::Unaligned>;

    const auto compute = [inner_dim, num_blocks, outer_block_size, buffer_data,
                          input_data, outer_dim](Eigen::Index start,
                                                 Eigen::Index limit) -> void {
      DCHECK(start >= 0 && limit <= num_blocks);
      int64 outer_dim_start = start * outer_block_size;
      int64 outer_dim_limit = limit * outer_block_size;
      outer_dim_limit = std::min(outer_dim, outer_dim_limit);

      Buffer buf(buffer_data + start * inner_dim, inner_dim);
      for (int64 i = outer_dim_start; i < outer_dim_limit; ++i) {
        auto in = Input(input_data + i * inner_dim, inner_dim);
        auto cast = in.template cast<AccumT>();
        buf = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf),
                                         const decltype(cast)>(buf, cast);
      }
    };

    // Compute cost of reducing a single block.
    const int64 compute_size = outer_block_size * inner_dim;
    const int64 compute_input_bytes = compute_size * sizeof(T);
    const Eigen::TensorOpCost cost(
        compute_input_bytes,
        0,  // We'll be mostly writing to L1, assume store cost is 0
        compute_size * Eigen::internal::functor_traits<BinaryFunctor>::Cost);

    device.parallelFor(num_blocks, cost, compute);

    // Aggregate partial results from temporary buffer into first block.
    auto buf0 = Buffer(buffer_data, inner_dim);
    // TODO(ezhulenev): Parallelize this loop for large inner dimensions?
    for (int i = 1; i < num_blocks; ++i) {
      auto buf = Buffer(buffer_data + i * inner_dim, inner_dim);
      buf0 = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf0),
                                        const decltype(buf)>(buf0, buf);
    }

    // Write final result to the output.
    output->template flat<T>() = buf0.template cast<T>();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUX_FUNCTOR_H_
