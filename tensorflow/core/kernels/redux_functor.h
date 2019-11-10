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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace functor {

// Compute reduction over outer dimensions.
// Example:
//   input: [D1, D2, ... , DN]
//   ->
//   output: [Di, ... , DN] where i belongs to set [1,N]
template <typename InputT, typename AccumT, typename OutputT,
          typename BinaryFunctor>
struct ReduceOuterDimensions {
  ReduceOuterDimensions() {}

  template <int num_dims>
  void operator()(const CPUDevice& device,
                  const Eigen::DSizes<Eigen::Index, num_dims>& input_dims,
                  const Tensor& input, Tensor* output) const {
    // Compute inner and outer dim after reshaping into 2d tensor.
    const int num_output_dims = output->dims();
    auto output_dims = output->template flat<OutputT>().dimensions();

    Eigen::Index inner_dim = 1, outer_dim = 1;
    for (int i = 0; i < num_dims - num_output_dims; ++i)
      outer_dim *= input_dims[i];
    for (int i = num_dims - num_output_dims; i < num_dims; ++i)
      inner_dim *= input_dims[i];

    if (1 == outer_dim) {
      // Nothing to do but passing input to output.
      output->template flat<OutputT>() =
          input.template flat<OutputT>().reshape(output_dims);
      return;
    }

    // Get device thread num.
    const Eigen::Index num_threads = device.numThreads();

    // If the inner dim parallelism is large enough
    // TODO(ezhulenev): There seems to be no benefits in going this route. Check
    // if this can be improved, or use better heuristic?
    if (inner_dim > num_threads * 32) {
      // Do not create more blocks than there are threads in a pool.
      const Eigen::Index num_blocks = num_threads;

      // Block size along the outer dimension.
      const Eigen::Index inner_block_size = Eigen::divup(inner_dim, num_blocks);
      const InputT* input_data = input.template flat<InputT>().data();

      // Allocate temporary buffer for partial reductions.
      Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index> buffer(
          {inner_dim});
      buffer.setZero();
      AccumT* buffer_data = buffer.data();

      using Buffer = Eigen::TensorMap<
          Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>,
          Eigen::Unaligned>;

      using Input = Eigen::TensorMap<
          Eigen::Tensor<const InputT, 1, Eigen::RowMajor, Eigen::Index>,
          Eigen::Unaligned>;

      const auto compute = [inner_dim, outer_dim, num_blocks, inner_block_size,
                            input_data, buffer_data](
                               Eigen::Index start, Eigen::Index limit) -> void {
        DCHECK(start >= 0 && limit <= num_blocks);
        Eigen::Index inner_dim_start = start * inner_block_size;
        Eigen::Index inner_dim_limit = limit * inner_block_size;
        inner_dim_limit = std::min(inner_dim, inner_dim_limit);
        Eigen::Index my_job_len = inner_dim_limit - inner_dim_start;

        const InputT* my_job_start = input_data + inner_dim_start;
        Buffer buf(buffer_data + inner_dim_start, my_job_len);

        for (Eigen::Index i = 0; i < outer_dim; ++i) {
          auto in = Input(my_job_start + i * inner_dim, my_job_len);
          auto cast = in.template cast<AccumT>();
          buf = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf),
                                           const decltype(cast)>(buf, cast);
        }
      };

      // Compute cost of reducing a single block.
      const Eigen::Index compute_size = outer_dim * inner_block_size;
      const Eigen::Index compute_input_bytes = compute_size * sizeof(InputT);
      const Eigen::TensorOpCost cost(
          compute_input_bytes,
          0,  // We'll be mostly writing to L1, assume store cost is 0
          compute_size * Eigen::internal::functor_traits<BinaryFunctor>::Cost);

      device.parallelFor(num_blocks, cost, compute);

      // Write final result to the output.
      output->template flat<OutputT>() =
          buffer.template cast<OutputT>().reshape(output_dims);
    } else {
      // Compute block size along the outer dimension for efficiency.
      const Eigen::Index parallel_cell_size = inner_dim;
      const Eigen::Index total_workload = outer_dim * inner_dim;
      const Eigen::Index max_parallelism = total_workload / parallel_cell_size;

      const Eigen::Index min_block_workload = 2000;
      const Eigen::Index min_block_size =
          Eigen::divup(min_block_workload, parallel_cell_size);
      const Eigen::Index max_num_blocks = std::min(
          max_parallelism, Eigen::divup(total_workload, min_block_size));

      // Do not create more blocks than there are threads in a pool.
      const Eigen::Index num_blocks = std::min(max_num_blocks, num_threads);

      // Block size along the outer dimension.
      const Eigen::Index outer_block_size = Eigen::divup(outer_dim, num_blocks);

      const InputT* input_data = input.template flat<InputT>().data();

      // Allocate temporary buffer for partial reductions.
      Tensor buffer(DataTypeToEnum<AccumT>::v(), {num_blocks, inner_dim});
      buffer.template flat<AccumT>().setZero();
      AccumT* buffer_data = buffer.template flat<AccumT>().data();

      using Buffer = Eigen::TensorMap<
          Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>,
          Eigen::Unaligned>;

      using Input = Eigen::TensorMap<
          Eigen::Tensor<const InputT, 1, Eigen::RowMajor, Eigen::Index>,
          Eigen::Unaligned>;

      const auto compute = [inner_dim, num_blocks, outer_block_size,
                            buffer_data, input_data, outer_dim](
                               Eigen::Index start, Eigen::Index limit) -> void {
        DCHECK(start >= 0 && limit <= num_blocks);
        Eigen::Index outer_dim_start = start * outer_block_size;
        Eigen::Index outer_dim_limit = limit * outer_block_size;
        outer_dim_limit = std::min(outer_dim, outer_dim_limit);

        Buffer buf(buffer_data + start * inner_dim, inner_dim);
        for (Eigen::Index i = outer_dim_start; i < outer_dim_limit; ++i) {
          auto in = Input(input_data + i * inner_dim, inner_dim);
          auto cast = in.template cast<AccumT>();
          buf = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf),
                                           const decltype(cast)>(buf, cast);
        }
      };

      // Compute cost of reducing a single block.
      const Eigen::Index compute_size = outer_block_size * inner_dim;
      const Eigen::Index compute_input_bytes = compute_size * sizeof(InputT);
      const Eigen::TensorOpCost cost(
          compute_input_bytes,
          0,  // We'll be mostly writing to L1, assume store cost is 0
          compute_size * Eigen::internal::functor_traits<BinaryFunctor>::Cost);

      device.parallelFor(num_blocks, cost, compute);

      // Aggregate partial results from temporary buffer into first block.
      auto buf0 = Buffer(buffer_data, inner_dim);
      // Just sum the buffer up, as inner dimensions is not large in this case.
      for (int i = 1; i < num_blocks; ++i) {
        auto buf = Buffer(buffer_data + i * inner_dim, inner_dim);
        buf0 = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf0),
                                          const decltype(buf)>(buf0, buf);
      }
      // Write final result to the output.
      output->template flat<OutputT>() =
          buf0.template cast<OutputT>().reshape(output_dims);
    }
  }
};

// Compute reduction to some serial middle dimensions (like a axis).
// Example:
//   input: [D1, D2, ... , DN]
//   ->
//   output: [Di, ... , Dj] where i & j belongs to set [1,N].
template <typename InputT, typename AccumT, typename OutputT,
          typename BinaryFunctor, typename Reducer>
struct ReduceMiddleDimensions {
  ReduceMiddleDimensions() {}

  template <int num_dims>
  void operator()(const CPUDevice& device,
                  const Eigen::DSizes<Eigen::Index, num_dims>& input_dims,
                  const Tensor& input, Tensor* output,
                  const int axis_begin_dim) const {
    // Compute dims after reshaping into 3d tensor.
    const int num_output_dims = output->dims();
    auto output_dims = output->template flat<OutputT>().dimensions();

    Eigen::Index inner_dim = 1, middle_dim = 1, outer_dim = 1;
    for (int i = 0; i < axis_begin_dim; ++i) outer_dim *= input_dims[i];
    for (int i = axis_begin_dim; i < axis_begin_dim + num_output_dims; ++i)
      middle_dim *= input_dims[i];
    for (int i = axis_begin_dim + num_output_dims; i < num_dims; ++i)
      inner_dim *= input_dims[i];

    if ((1 == inner_dim * outer_dim)) {
      // Nothing to do.
      output->template flat<OutputT>() =
          input.template flat<OutputT>().reshape(output_dims);
      return;
    } else if (1 == inner_dim) {
      // Equivalent to ReduceOuterDimensions.
      const ReduceOuterDimensions<InputT, AccumT, OutputT, BinaryFunctor> redux;
      redux(device, input_dims, input, output);
      return;
    }

    // Compute block size along the outer dimension for efficiency.
    const Eigen::Index parallel_cell_size = inner_dim;
    const Eigen::Index max_parallelism = outer_dim * middle_dim;
    const Eigen::Index total_workload = max_parallelism * inner_dim;

    const Eigen::Index min_block_workload = 2000;
    const Eigen::Index min_block_size =
        Eigen::divup(min_block_workload, parallel_cell_size);
    const Eigen::Index max_num_blocks =
        std::min(max_parallelism, Eigen::divup(total_workload, min_block_size));

    // Do not create more blocks than there are threads in a pool.
    const Eigen::Index num_threads = device.numThreads();
    const Eigen::Index num_blocks = std::min(max_num_blocks, num_threads);

    // Block size along the outer dimension.
    const Eigen::Index outer_block_size =
        Eigen::divup(total_workload, num_blocks);

    const InputT* input_data = input.template flat<InputT>().data();

    // Allocate temporary buffer for partial reductions.
    Eigen::Tensor<AccumT, 2> buffer(num_blocks, middle_dim);
    buffer.setZero();
    AccumT* buffer_data = buffer.data();

    using Buffer = Eigen::TensorMap<Eigen::Tensor<AccumT, 1>>;
    using Input = Eigen::TensorMap<Eigen::Tensor<const InputT, 1>>;

    Eigen::array<Eigen::Index, 1> reduction_axis = {0};
    Reducer reducer;
    const BinaryFunctor binary_op;

    const auto compute = [inner_dim, middle_dim, input_data, buffer_data,
                          total_workload, num_blocks, outer_block_size,
                          reduction_axis, reducer, binary_op](
                             Eigen::Index start, Eigen::Index limit) -> void {
      DCHECK(start >= 0 && limit <= num_blocks);
      Eigen::Index block_start = start * outer_block_size;
      Eigen::Index block_limit = limit * outer_block_size;
      block_limit = std::min(total_workload, block_limit);
      Buffer buf(buffer_data + start * middle_dim, middle_dim);

      const int align_start =
          ((block_start + inner_dim - 1) / inner_dim) * inner_dim;
      const int align_end = (block_limit / inner_dim) * inner_dim;

      Eigen::Index coordinate = block_start / inner_dim % middle_dim;
      Eigen::Tensor<AccumT, 0> reduced =
          Input(&input_data[block_start], align_start - block_start)
              .reduce(reduction_axis, reducer)
              .template cast<AccumT>();

      buf(coordinate) = binary_op(buf(coordinate), reduced(0));

      coordinate = align_start / inner_dim % middle_dim;
      for (int i = align_start; i < align_end; i += inner_dim) {
        reduced = Input(&input_data[i], inner_dim)
                      .reduce(reduction_axis, reducer)
                      .template cast<AccumT>();
        buf(coordinate) = binary_op(buf(coordinate), reduced(0));
        ++coordinate;
        if (middle_dim == coordinate) coordinate = 0;
      }

      reduced = Input(&input_data[align_end], block_limit - align_end)
                    .reduce(reduction_axis, reducer)
                    .template cast<AccumT>();
      buf(coordinate) = binary_op(buf(coordinate), reduced(0));
    };

    // Compute cost of reducing a single block.
    const Eigen::Index compute_size = outer_block_size * inner_dim;
    const Eigen::Index compute_input_bytes = compute_size * sizeof(InputT);
    const Eigen::TensorOpCost cost(
        compute_input_bytes,
        0,  // We'll be mostly writing to L1, assume store cost is 0
        compute_size * Eigen::internal::functor_traits<BinaryFunctor>::Cost);

    device.parallelFor(num_blocks, cost, compute);

    using Output = Eigen::TensorMap<
        Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>,
        Eigen::Unaligned>;
    // Aggregate partial results from temporary buffer into first block.
    auto buf0 = Output(buffer_data, middle_dim);
    // TODO(ezhulenev): Parallelize this loop for large inner dimensions?
    for (int i = 1; i < num_blocks; ++i) {
      auto buf = Output(buffer_data + i * middle_dim, middle_dim);
      buf0 = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf0),
                                        const decltype(buf)>(buf0, buf);
    }

    // Write final result to the output.
    output->template flat<OutputT>() =
        buf0.template cast<OutputT>().reshape(output_dims);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUX_FUNCTOR_H_
