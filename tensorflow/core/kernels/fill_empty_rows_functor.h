/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FILL_EMPTY_ROWS_OP_H_
#define TENSORFLOW_CORE_KERNELS_FILL_EMPTY_ROWS_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename Tindex, bool RaggedOperands>
struct FillEmptyRows {
  // Note that the done callback is only used by the GPU implementation.
  absl::Status operator()(OpKernelContext* context,
                          const Tensor& default_value_t,
                          const Tensor& indices_t, const Tensor& values_t,
                          const Tensor& dense_shape_t,
                          typename AsyncOpKernel::DoneCallback done = nullptr);
};

template <typename T, typename Tindex, bool RaggedOperands>
struct FillEmptyRows<CPUDevice, T, Tindex, RaggedOperands> {
  static constexpr int IndicesRank = RaggedOperands ? 1 : 2;
  absl::Status operator()(OpKernelContext* context,
                          const Tensor& default_value_t,
                          const Tensor& indices_t, const Tensor& values_t,
                          const Tensor& dense_shape_t,
                          typename AsyncOpKernel::DoneCallback done) {
    (void)done;  // Unused (only used in GPU implementation)
    const int kOutputIndicesOutput = 0;
    const int kOutputValuesOutput = 1;
    const int kEmptyRowIndicatorOutput = 2;
    const int kReverseIndexMapOutput = 3;

    const T& default_value = default_value_t.scalar<T>()();
    const auto indices = indices_t.tensor<Tindex, IndicesRank>();
    const auto values = values_t.vec<T>();
    const auto dense_shape = dense_shape_t.tensor<Tindex, IndicesRank - 1>();

    const Tindex N = indices_t.shape().dim_size(0);
    const Tindex dense_rows = dense_shape(0);

    bool* empty_row_indicator = nullptr;
    if (context->output_required(kEmptyRowIndicatorOutput)) {
      Tensor* empty_row_indicator_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({dense_rows}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kEmptyRowIndicatorOutput, output_shape, &empty_row_indicator_t));
      empty_row_indicator = empty_row_indicator_t->vec<bool>().data();
    }
    Tindex* reverse_index_map = nullptr;
    if (context->output_required(kReverseIndexMapOutput)) {
      Tensor* reverse_index_map_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape({N}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kReverseIndexMapOutput, output_shape, &reverse_index_map_t));
      reverse_index_map = reverse_index_map_t->vec<Tindex>().data();
    }

    const int rank = IndicesRank == 1 ? 1 : indices_t.shape().dim_size(1);

    if (dense_rows == 0) {
      if (N != 0) {
        return errors::InvalidArgument(
            "Received SparseTensor with dense_shape[0] = 0 but "
            "indices.shape[0] = ",
            N);
      }
      Tensor* output_indices_t;
      TensorShape output_indices_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({0, rank}, &output_indices_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputIndicesOutput, output_indices_shape, &output_indices_t));
      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({0}), &output_values_t));

      // Exit early, nothing more to do.
      return absl::OkStatus();
    }

    auto vec_or_matrix = [](auto tensor, int index1, int index2) -> auto& {
      std::array<int, IndicesRank> indices;
      indices[0] = index1;
      if (IndicesRank == 2) {
        indices[1] = index2;
      }
      return std::apply(tensor, indices);
    };

    bool rows_are_ordered = true;
    Tindex last_indices_row = 0;
    std::vector<Tindex> csr_offset(dense_rows, 0);
    for (int i = 0; i < N; ++i) {
      const Tindex row = vec_or_matrix(indices, i, 0);
      if (row < 0 || row >= dense_rows) {
        return errors::InvalidArgument("indices(", i, ", 0) is invalid: ", row,
                                       " >= ", dense_rows);
      }
      ++csr_offset[row];
      rows_are_ordered = rows_are_ordered & (row >= last_indices_row);
      last_indices_row = row;
    }
    bool all_rows_full = true;
    for (int row = 0; row < dense_rows; ++row) {
      // csr_offset here describes the number of elements in this dense row
      bool row_empty = (csr_offset[row] == 0);
      if (empty_row_indicator) {
        empty_row_indicator[row] = row_empty;
      }
      all_rows_full = all_rows_full & !row_empty;
      // In filled version, each row has at least one element.
      csr_offset[row] = std::max(csr_offset[row], Tindex{1});
      // Update csr_offset to represent the number of elements up to and
      // including dense_row + 1:
      //  csr_offset(0) == #{elements of row 0}
      //  csr_offset(1) == #{elements of row 1} + #{elements of row 0}
      //  ..
      //  csr_offset(i) == starting index for elements in row i + 1.
      if (row > 0) {
        csr_offset[row] += csr_offset[row - 1];
      }
    }

    if (all_rows_full && rows_are_ordered) {
      context->set_output(kOutputIndicesOutput, indices_t);
      context->set_output(kOutputValuesOutput, values_t);
      if (reverse_index_map) {
        for (Tindex i = 0; i < N; ++i) {
          reverse_index_map[i] = i;
        }
      }
    } else {
      Tensor* output_indices_t;
      const Tindex N_full = csr_offset[dense_rows - 1];
      TensorShape output_indices_shape;
      if constexpr (RaggedOperands) {
        TF_RETURN_IF_ERROR(
            TensorShape::BuildTensorShape({N_full}, &output_indices_shape));
      } else {
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(
            {N_full, rank}, &output_indices_shape));
      }
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputIndicesOutput, output_indices_shape, &output_indices_t));
      auto output_indices = output_indices_t->tensor<Tindex, IndicesRank>();

      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({N_full}), &output_values_t));
      auto output_values = output_values_t->vec<T>();

      std::vector<Tindex> filled_count(dense_rows, 0);

      // Fill in values for rows that are not missing
      for (Tindex i = 0; i < N; ++i) {
        const Tindex row = vec_or_matrix(indices, i, 0);
        Tindex& offset = filled_count[row];
        const Tindex output_i = ((row == 0) ? 0 : csr_offset[row - 1]) + offset;
        offset++;  // Increment the filled count for this row.
        std::copy_n(&vec_or_matrix(indices, i, 0), rank,
                    &vec_or_matrix(output_indices, output_i, 0));
        output_values(output_i) = values(i);
        // We'll need this reverse index map to backprop correctly.
        if (reverse_index_map) {
          reverse_index_map[i] = output_i;
        }
      }

      // Fill in values for rows that are missing
      for (Tindex row = 0; row < dense_rows; ++row) {
        const Tindex row_count = filled_count[row];
        if (row_count == 0) {  // We haven't filled this row
          const Tindex starting_index = (row == 0) ? 0 : csr_offset[row - 1];
          // Remaining index values were set to zero already.
          // Just need to set the row index in the right location.
          vec_or_matrix(output_indices, starting_index, 0) = row;
          for (Tindex col = 1; col < rank; ++col) {
            vec_or_matrix(output_indices, starting_index, col) = 0;
          }
          output_values(starting_index) = default_value;
        }
      }
    }

    return absl::OkStatus();
  }
};

template <typename Device, typename T, typename Tindex>
struct FillEmptyRowsGrad {
  absl::Status operator()(OpKernelContext* context,
                          typename TTypes<Tindex>::ConstVec reverse_index_map,
                          typename TTypes<T>::ConstVec grad_values,
                          typename TTypes<T>::Vec d_values,
                          typename TTypes<T>::Scalar d_default_value);
};

template <typename T, typename Tindex>
struct FillEmptyRowsGrad<CPUDevice, T, Tindex> {
  absl::Status operator()(OpKernelContext* context,
                          typename TTypes<Tindex>::ConstVec reverse_index_map,
                          typename TTypes<T>::ConstVec grad_values,
                          typename TTypes<T>::Vec d_values,
                          typename TTypes<T>::Scalar d_default_value) {
    const CPUDevice& device = context->eigen_device<CPUDevice>();
    const Tindex N = reverse_index_map.dimension(0);
    const Tindex N_full = grad_values.dimension(0);

    T& d_default_value_scalar = d_default_value();
    d_default_value_scalar = T();

    Tensor visited_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(device) = visited.constant(false);

    for (int i = 0; i < N; ++i) {
      // Locate the index of the output of the forward prop associated
      // with this location in the input of the forward prop.  Copy
      // the gradient into it.  Mark it as visited.
      int64_t reverse_index = reverse_index_map(i);
      if (reverse_index < 0 || reverse_index >= N_full) {
        return errors::InvalidArgument(
            "Elements in reverse index must be in [0, ", N_full, ") but got ",
            reverse_index);
      }
      d_values(i) = grad_values(reverse_index);
      visited(reverse_index) = true;
    }
    for (int j = 0; j < N_full; ++j) {
      // The default value gradient gets the accumulated remainder of
      // the backprop values (since the default value was used to fill
      // in these slots in the forward calculation).
      if (!visited(j)) {
        d_default_value_scalar += grad_values(j);
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FILL_EMPTY_ROWS_OP_H_
