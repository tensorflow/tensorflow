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

#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/Eigen/Core"

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

template <typename T>
class CholeskyGrad : public OpKernel {
 public:
  explicit CholeskyGrad(OpKernelConstruction* context) : OpKernel(context) {}
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using ConstRef = Eigen::Ref<const Matrix>;
  using Ref = Eigen::Ref<Matrix>;

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor_l = context->input(0);
    const Tensor& input_tensor_grad = context->input(1);
    // Check that input tensors represent a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_l.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_grad.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    // Check that input tensors are square.
    OP_REQUIRES(context,
                input_tensor_l.dim_size(0) == input_tensor_l.dim_size(1),
                errors::InvalidArgument("Input matrix must be square."));
    OP_REQUIRES(context,
                input_tensor_grad.dim_size(0) == input_tensor_grad.dim_size(1),
                errors::InvalidArgument("Input matrix must be square."));

    // Check that input tensors are of same size.
    OP_REQUIRES(context,
                input_tensor_l.dim_size(0) == input_tensor_grad.dim_size(0),
                errors::InvalidArgument("Input matrices must be same size."));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, input_tensor_grad.shape(), &output_tensor));

    if (output_tensor->NumElements() == 0) {
      // the output shape is a 0-element matrix, so there is nothing to do.
      return;
    }
    // The next lines are necessary to get Eigen matrix behaviour.
    const ConstMatrixMap input_matrix_l_full(input_tensor_l.flat<T>().data(),
                                             input_tensor_l.dim_size(0),
                                             input_tensor_l.dim_size(1));
    const ConstMatrixMap input_matrix_grad(input_tensor_grad.flat<T>().data(),
                                           input_tensor_grad.dim_size(0),
                                           input_tensor_grad.dim_size(1));
    MatrixMap output_matrix(output_tensor->template flat<T>().data(),
                            input_tensor_l.dim_size(0),
                            input_tensor_l.dim_size(1));

    // Algorithm only depends on lower triangular half on input_tensor_l.
    const Matrix input_matrix_l =
        input_matrix_l_full.template triangularView<Eigen::Lower>();
    // Algorithm only depends on lower triangular half on input_matrix_grad.
    output_matrix = input_matrix_grad.template triangularView<Eigen::Lower>();

    const int64 kMatrixSize = input_matrix_l.rows();
    const int64 kMaxBlockSize = 32;

    for (int64 block_end = kMatrixSize; block_end > 0;
         block_end -= kMaxBlockSize) {
      /* This shows the block structure.

      /      \
      |      |
      | R D  |
      \ B C  /

      Variables names representing the derivative matrix have a trailing '_bar'.
      */

      const int64 block_begin = std::max(0ll, block_end - kMaxBlockSize);
      const int64 block_size = block_end - block_begin;
      const int64 trailing_size = kMatrixSize - block_end;

      auto B = input_matrix_l.block(block_end, 0, trailing_size, block_begin);
      auto B_bar =
          output_matrix.block(block_end, 0, trailing_size, block_begin);

      auto C = input_matrix_l.block(block_end, block_begin, trailing_size,
                                    block_size);
      auto C_bar = output_matrix.block(block_end, block_begin, trailing_size,
                                       block_size);

      auto D = input_matrix_l.block(block_begin, block_begin, block_size,
                                    block_size);
      auto D_bar =
          output_matrix.block(block_begin, block_begin, block_size, block_size);

      auto R = input_matrix_l.block(block_begin, 0, block_size, block_begin);
      auto R_bar = output_matrix.block(block_begin, 0, block_size, block_begin);

      C_bar = D.adjoint().template triangularView<Eigen::Upper>()
          .solve(C_bar.adjoint()).adjoint();
      D_bar -= (C_bar.adjoint() * C).template triangularView<Eigen::Lower>();
      B_bar -= C_bar * R;
      R_bar -= C_bar.adjoint() * B;
      CholeskyGradUnblocked(D, D_bar);
      R_bar -= (D_bar + D_bar.adjoint()) * R;
    }
    output_matrix = (0.5 * (output_matrix + output_matrix.transpose())).eval();
  }
  void CholeskyGradUnblocked(const ConstRef l_block, Ref grad_block) {
    const int64 kMatrixSize = l_block.rows();
    for (int64 k = kMatrixSize - 1; k >= 0; k--) {
      /* This shows the block structure.

      /      \
      |      |
      | r d  |
      \ B c  /

      Variables names representing the derivative matrix have a trailing '_bar'.
      */

      const int64 number_rows_B = kMatrixSize - (k + 1);
      const int64 number_rows_r_stack_B = number_rows_B + 1;

      auto r = l_block.block(k, 0, 1, k);
      auto r_bar = grad_block.block(k, 0, 1, k);
      auto d = l_block(k, k);  // This needs to be a scalar rather than a view.
      auto d_bar = grad_block.block(k, k, 1, 1);
      // B is not included explicitly because it is not used on its own.
      auto B_bar = grad_block.block(k + 1, 0, number_rows_B, k);
      auto c = l_block.block(k + 1, k, number_rows_B, 1);
      auto c_bar = grad_block.block(k + 1, k, number_rows_B, 1);
      // Result of vertical stacking d_bar and c_bar.
      auto d_stack_c_bar = grad_block.block(k, k, number_rows_r_stack_B, 1);
      // Result of vertical stacking of r and B.
      auto r_stack_B = l_block.block(k, 0, number_rows_r_stack_B, k);
      d_bar -= (c.adjoint() * c_bar) / d;
      d_stack_c_bar /= d;
      r_bar -= d_stack_c_bar.adjoint() * r_stack_B;
      B_bar -= c_bar * r;
      d_bar /= 2.;
    }
  }
};

REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<float>), float);
REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<double>), double);
}  // namespace tensorflow
