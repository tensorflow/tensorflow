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

#ifndef TENSORFLOW_KERNELS_KERNELS_BINARY_LINALG_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_KERNELS_BINARY_LINALG_OPS_COMMON_H_

// Classes to support binary linear algebra operations. This should eventually
// be merged into third_party/tensorflow/core/kernels/linalg_ops_common.h.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Base class for binary linear algebra operators.
class BinaryLinearAlgebraOpBase : public OpKernel {
 public:
  explicit BinaryLinearAlgebraOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}
  ~BinaryLinearAlgebraOpBase() override {}

  // Return the output shape of each individual matrix operation. Must be
  // rank 0, 1, or 2.  Scalar outputs are rank 0.
  virtual TensorShape GetOutputMatrixShape(
      const TensorShape& in_lhs_matrix_shape,
      const TensorShape& in_rhs_matrix_shape) = 0;

  // Return the cost per matrix operation. Cost per unit is assumed to be
  // roughly 1ns, based on comments in core/util/work_sharder.cc.
  virtual int64 GetCostPerUnit(const TensorShape& in_lhs_matrix_shape,
                               const TensorShape& in_rhs_matrix_shape) = 0;

  // If SupportsBatchOperation() returns false, this Op will only accept rank 2
  // (if the supported input type is a matrix). If it returns true, the Op will
  // accept inputs of rank >= 3, and repeatedly execute the operation on all
  // matrices in the innermost two dimensions.
  virtual bool SupportsBatchOperation() = 0;

  // Perform the actual computation on an input matrix, and store the results
  // in the output. This will be called repeatedly for a single call to
  // Compute(), if multiple matrices exist in the input Tensor.
  //
  // This function should only compute the results for a single input matrix.
  // The 'matrix_index' parameter specifies the index of the matrix to be used
  // from the input, and the index of the matrix to be written to in the output.
  // The two input matrices are in row major order, and located at the memory
  // addresses
  //   a_in.flat<Scalar>().data() +
  //   matrix_index * a_in_matrix_shape.num_elements(), and
  //   b_in.flat<Scalar>().data() +
  //   matrix_index * b_in_matrix_shape.num_elements().
  // The output matrix is in row major order, and is located at the memory
  // address
  //   out->flat<Scalar>().data() +
  //   matrix_index * output_matrix_shape.num_elements().
  // The BinaryLinearAlgebraOp<Scalar> class below has functionality which
  // performs
  // this mapping and presents an interface based on the Eigen::MatrixBase API.
  virtual void ComputeMatrix(OpKernelContext* context, int64 matrix_index,
                             const Tensor& a_in,
                             const TensorShape& a_in_matrix_shape,
                             const Tensor& b_in,
                             const TensorShape& b_in_matrix_shape,
                             Tensor* output,
                             const TensorShape& output_matrix_shape) = 0;
  void Compute(OpKernelContext* context) override;
};

// This base class encapsulates the functionality of mapping the input and
// output tensors using Eigen::Map, so that the Eigen::MatrixBase API may be
// directly used by derived classes.
// SupportsBatchOperationT is a bool template argument which if set to true
// will allow the Op to process batches of matrices (rank >= 3); if set to
// false the Op will only accept rank 2 inputs.
template <typename Scalar, bool SupportsBatchOperationT>
class BinaryLinearAlgebraOp : public BinaryLinearAlgebraOpBase {
 public:
  explicit BinaryLinearAlgebraOp(OpKernelConstruction* context)
      : BinaryLinearAlgebraOpBase(context) {}

  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap =
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>;
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Perform the actual computation on the input matrix, and store the results
  // in the output. This will be called repeatedly for a single call to
  // Compute(), if multiple matrices exist in the input Tensor.
  virtual void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& a,
                             const ConstMatrixMap& b, MatrixMap* output) = 0;

  bool SupportsBatchOperation() final { return SupportsBatchOperationT; }

  // A concrete implementation of BinaryLinearAlgebraOpBase::ComputeMatrix().
  void ComputeMatrix(OpKernelContext* context, int64 matrix_index,
                     const Tensor& a_in, const TensorShape& a_in_matrix_shape,
                     const Tensor& b_in, const TensorShape& b_in_matrix_shape,
                     Tensor* output,
                     const TensorShape& output_matrix_shape) final;
};

// Declare that BinaryLinearAlgebraOp is explicitly instantiated in
// linalg_ops_common.cc for float and double.
extern template class BinaryLinearAlgebraOp<float, false>;
extern template class BinaryLinearAlgebraOp<float, true>;
extern template class BinaryLinearAlgebraOp<double, false>;
extern template class BinaryLinearAlgebraOp<double, true>;

}  // namespace tensorflow

#define REGISTER_BINARY_LINALG_OP(OpName, OpClass, Scalar) \
  REGISTER_KERNEL_BUILDER(                                 \
      Name(OpName).Device(DEVICE_CPU).TypeConstraint<Scalar>("T"), OpClass)

#endif  // TENSORFLOW_KERNELS_KERNELS_BINARY_LINALG_OPS_COMMON_H_
