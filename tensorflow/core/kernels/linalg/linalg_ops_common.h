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
#ifndef TENSORFLOW_CORE_KERNELS_LINALG_LINALG_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_LINALG_OPS_COMMON_H_

// Classes to support linear algebra functionality, similar to the numpy.linalg
// module. Supports batch computation on several matrices at once, sharding the
// computations across different threads if necessary.
#include <algorithm>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Base class for linear algebra operators.
template <class InputScalar, class OutputScalar = InputScalar>
class LinearAlgebraOp : public OpKernel {
 public:
  explicit LinearAlgebraOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override;

 protected:
  using TensorShapes = gtl::InlinedVector<TensorShape, 4>;
  // Returns the number of leading inputs that are to be treated as matrix
  // inputs. By default this is all the inputs. Derived classes can override
  // this to tell the base class to ignore one or more trailing inputs.
  virtual int NumMatrixInputs(const OpKernelContext* context) const {
    return context->num_inputs();
  }

  // Returns true if the number of inputs and their shapes are as expected.
  // Many ops take a single square input matrix, so we provide that as a default
  // implementation for convenience.
  virtual void ValidateInputMatrixShapes(
      OpKernelContext* context, const TensorShapes& input_matrix_shapes) const {
    ValidateSingleSquareMatrix(context, input_matrix_shapes);
  }

  // Convenience validators for common cases:
  //
  // Validate op taking a single matrix A.
  static void ValidateSingleMatrix(OpKernelContext* context,
                                   const TensorShapes& input_matrix_shapes);
  // Validate op taking a single square matrix A.
  static void ValidateSingleSquareMatrix(
      OpKernelContext* context, const TensorShapes& input_matrix_shapes);
  // Validate op taking two matrices A and B that have the same number of rows.
  static void ValidateSolver(OpKernelContext* context,
                             const TensorShapes& input_matrix_shapes);
  // Validate op taking two matrices A and B that have the same number of rows
  // and A is square.
  static void ValidateSquareSolver(OpKernelContext* context,
                                   const TensorShapes& input_matrix_shapes);

  // Returns the output shapes of each individual matrix operation. Output
  // matrices shapes must be rank 0, 1, or 2. Scalar outputs are rank 0.
  //
  // The derived class may return a number of shapes (N) less than
  // context->num_outputs() (M) to indicate that a only leading subset of
  // the outputs will be populated. In this case, a dummy scalar tensor with
  // value zero will be return for the last M-N outputs.
  //
  // For many ops, the output dimensions are the same as the input dimensions,
  // so we provide that as a default implementation for convenience.
  virtual TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const {
    return input_matrix_shapes;
  }

  // Returns the cost per matrix operation. This is used to determine the
  // number of threads to use for parallelizing calls to ComputeMatrix in
  // batch mode. Cost per unit is assumed to be roughly 1ns, based on comments
  // in core/util/work_sharder.cc. Many linear algebra ops take roughly max(m,n)
  // * min(m,n)^2, where the first input matrix is m-by-n. We provide that as a
  // default implementation for convenience.
  virtual int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double cost = std::max(m, n) * std::min(m, n) * std::min(m, n);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  // Returns true if it is safe to forward (alias) input to output buffer
  // and expect the kernel to perform the computation inplace.
  virtual bool EnableInputForwarding() const { return true; }

  using InputMatrix = Eigen::Matrix<InputScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>;
  using InputConstMatrixMap = Eigen::Map<const InputMatrix>;
  using InputMatrixMap = Eigen::Map<InputMatrix>;
  using InputConstVectorMap =
      Eigen::Map<const Eigen::Matrix<InputScalar, 1, Eigen::Dynamic>>;
  using InputConstMatrixMaps = gtl::InlinedVector<InputConstMatrixMap, 4>;
  using InputMatrixMaps = gtl::InlinedVector<InputMatrixMap, 4>;
  using InputRealScalar = typename Eigen::NumTraits<InputScalar>::Real;

  using OutputMatrix = Eigen::Matrix<OutputScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
  using OutputConstMatrixMap = Eigen::Map<const OutputMatrix>;
  using OutputMatrixMap = Eigen::Map<OutputMatrix>;
  using OutputConstVectorMap =
      Eigen::Map<const Eigen::Matrix<OutputScalar, 1, Eigen::Dynamic>>;
  using OutputConstMatrixMaps = gtl::InlinedVector<OutputConstMatrixMap, 4>;
  using OutputMatrixMaps = gtl::InlinedVector<OutputMatrixMap, 4>;
  using OutputRealScalar = typename Eigen::NumTraits<OutputScalar>::Real;

  // backward compatibility
  using Scalar = OutputScalar;
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using ConstVectorMap =
      Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>;
  using ConstMatrixMaps = gtl::InlinedVector<ConstMatrixMap, 4>;
  using MatrixMaps = gtl::InlinedVector<MatrixMap, 4>;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  // Performs a single matrix computation given input matrices, and
  // stores the result in outputs. For batch operations, this will be called
  // repeatedly for a single call to Compute() when multiple matrices exist in
  // input Tensors with rank > 2. In this case the calls to ComputeMatrix are
  // parallelized. The number of threads used is determined by a cost model from
  // the value returned by GetCostPerUnit().
  virtual void ComputeMatrix(OpKernelContext* context,
                             const InputConstMatrixMaps& inputs,
                             OutputMatrixMaps* outputs) = 0;

 private:
  using TensorInputs = gtl::InlinedVector<const Tensor*, 4>;
  using TensorOutputs = gtl::InlinedVector<Tensor*, 4>;
  // This function maps 2-d slices (matrices) of the input and output tensors
  // using Eigen::Map and calls ComputeMatrix implemented in terms of the
  // Eigen::MatrixBase API by the derived class.
  //
  // The 'matrix_index' parameter specifies the index of the matrix to be used
  // from each input tensor, and the index of the matrix to be written to each
  // output tensor. The input matrices are in row major order, and located at
  // the memory addresses
  //   inputs[i].flat<Scalar>().data() +
  //   matrix_index * input_matrix_shapes[i].num_elements()
  // for i in 0...inputs.size()-1.
  // The output matrices are in row major order, and located at the memory
  // address
  //   outputs[i]->flat<Scalar>().data() +
  //   matrix_index * output_matrix_shapes[i].num_elements().
  // for i in 0...outputs.size()-1.
  //
  void ComputeTensorSlice(OpKernelContext* context, int64 matrix_index,
                          const TensorInputs& inputs,
                          const TensorShapes& input_matrix_shapes,
                          const TensorOutputs& outputs,
                          const TensorShapes& output_matrix_shapes);

  void AnalyzeInputs(OpKernelContext* context, TensorInputs* inputs,
                     TensorShapes* input_matrix_shapes,
                     TensorShape* batch_shape);

  void PrepareOutputs(OpKernelContext* context,
                      const TensorShapes& input_matrix_shapes,
                      const TensorShape& batch_shape, TensorOutputs* outputs,
                      TensorShapes* output_matrix_shapes);
};

// Declare LinearAlgebraOp, which is explicitly instantiated in
// linalg_ops_common.cc for float, double, complex64, and complex128.
extern template class LinearAlgebraOp<float>;
extern template class LinearAlgebraOp<double>;
extern template class LinearAlgebraOp<complex64>;
extern template class LinearAlgebraOp<complex128>;

}  // namespace tensorflow

#define INHERIT_LINALG_TYPEDEFS(Scalar)                       \
  typedef LinearAlgebraOp<Scalar> Base;                       \
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real; \
  using Matrix = typename Base::Matrix;                       \
  using MatrixMap = typename Base::MatrixMap;                 \
  using MatrixMaps = typename Base::MatrixMaps;               \
  using ConstMatrixMap = typename Base::ConstMatrixMap;       \
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;     \
  using ConstVectorMap = typename Base::ConstVectorMap;       \
  using TensorShapes = typename Base::TensorShapes;

#define REGISTER_LINALG_OP_CPU(OpName, OpClass, Scalar) \
  REGISTER_KERNEL_BUILDER(                              \
      Name(OpName).Device(DEVICE_CPU).TypeConstraint<Scalar>("T"), OpClass)

#define REGISTER_LINALG_OP_GPU(OpName, OpClass, Scalar) \
  REGISTER_KERNEL_BUILDER(                              \
      Name(OpName).Device(DEVICE_GPU).TypeConstraint<Scalar>("T"), OpClass)

// Deprecated, use one of the device-specific macros above.
#define REGISTER_LINALG_OP(OpName, OpClass, Scalar) \
  REGISTER_LINALG_OP_CPU(OpName, OpClass, Scalar)

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_LINALG_OPS_COMMON_H_
