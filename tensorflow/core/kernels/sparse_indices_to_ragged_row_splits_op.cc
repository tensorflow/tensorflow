/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sparse_indices_to_ragged_row_splits_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace functor {

template <typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor<CPUDevice, IndexType> {
  Status operator()(OpKernelContext* context, int num_nonzero,
                    bool validate_ragged_right,
                    const IndexType* indices_flat_2d,
                    const IndexType* dense_shape, int32_t* invalid_flag) {
    auto num_rows = dense_shape[0];
    auto num_cols = dense_shape[1];

    Tensor* output;
    TF_RETURN_IF_ERROR(context,
                       context->allocate_output(
                           "row_splits", TensorShape({num_rows + 1}), &output));
    IndexType* row_splits = output->flat<IndexType>().data();

    *invalid_flag = 0;

    int prev_row = -1;
    int prev_col = -1;
    int n = 0;
    for (; n < num_nonzero; ++n) {
      int curr_row = indices_flat_2d[2 * n];
      if (validate_ragged_right) {
        if (curr_row != prev_row) {
          // When the current indices represent the start of a new row,
          // check that the row index increases monotonically and does not
          // exceed the dense size.
          // This ensures indices are in order by row.
          if ((curr_row < prev_row) || (curr_row >= num_rows)) {
            *invalid_flag = 1;
            return OkStatus();
          }
          prev_col = -1;
        }
        // Check that each column index within a row is one higher than the
        // previous and that the first column index is 0 (prev_col + 1, where
        // prev_col = -1). This ensures that the tensor is ragged-right and
        // indices are in order.
        int curr_col = indices_flat_2d[2 * n + 1];
        if ((curr_col != prev_col + 1) || (curr_col >= num_cols)) {
          *invalid_flag = 1;
          return OkStatus();
        } else {
          prev_col = curr_col;
        }
      }
      // Fill in row_splits with the current index. A loop is used to fill all
      // entries if there are empty rows between the previous and current index.
      for (int r = prev_row; r < curr_row; ++r) {
        row_splits[r + 1] = n;
      }
      prev_row = curr_row;
    }
    // Fill in the final row split and those for any trailing empty rows.
    for (int r = prev_row; r < num_rows; ++r) {
      row_splits[r + 1] = n;
    }

    return OkStatus();
  }
};

}  // namespace functor

template <typename Device, typename IndexType>
class SparseIndicesToRaggedRowSplitsOp : public OpKernel {
 public:
  explicit SparseIndicesToRaggedRowSplitsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("validate_ragged_right",
                                             &validate_ragged_right_));
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& dense_shape = context->input(1);

    auto num_nonzero = indices.dim_size(0);

    Tensor* output_invalid = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("invalid_flag", TensorShape({}),
                                            &output_invalid));

    OP_REQUIRES_OK(
        context,
        functor::SparseIndicesToRaggedRowSplitsFunctor<Device, IndexType>()(
            context, num_nonzero, validate_ragged_right_,
            indices.flat<IndexType>().data(),
            dense_shape.flat<IndexType>().data(),
            output_invalid->flat<int32_t>().data()));
  }

 private:
  bool validate_ragged_right_;
};

#define REGISTER_CPU(IndexType)                    \
  REGISTER_KERNEL_BUILDER(                         \
      Name("SparseIndicesToRaggedRowSplits")       \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<IndexType>("IndexType"), \
      SparseIndicesToRaggedRowSplitsOp<CPUDevice, IndexType>)
REGISTER_CPU(int32);
REGISTER_CPU(int64);
#undef REGISTER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using GPUDevice = Eigen::GpuDevice;

#define REGISTER_GPU(IndexType)                    \
  REGISTER_KERNEL_BUILDER(                         \
      Name("SparseIndicesToRaggedRowSplits")       \
          .Device(DEVICE_GPU)                      \
          .HostMemory("dense_shape")               \
          .TypeConstraint<IndexType>("IndexType"), \
      SparseIndicesToRaggedRowSplitsOp<GPUDevice, IndexType>)
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
