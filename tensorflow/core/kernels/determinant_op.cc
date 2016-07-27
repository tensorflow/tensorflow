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

// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperation>
class DeterminantOp : public LinearAlgebraOp<Scalar, SupportsBatchOperation> {
 public:
  typedef LinearAlgebraOp<Scalar, SupportsBatchOperation> Base;

  explicit DeterminantOp(OpKernelConstruction* context) : Base(context) {}

  using TensorShapes = typename Base::TensorShapes;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shape) const final {
    return TensorShapes({TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Scalar determinant;
    if (inputs[0].rows() == 0) {
      // An empty matrix' determinant is defined to be 1.  See wikipedia.
      determinant = 1;
    } else {
      determinant = inputs[0].determinant();
    }
    // TODO(rmlarsen): Don't fail on infinite determinants, since that could
    // be a valid result and the user should check for it instead.
    OP_REQUIRES(context, Eigen::numext::isfinite(determinant),
                errors::InvalidArgument("The determinant is not finite."));
    outputs->at(0)(0, 0) = determinant;
  }
};

REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<float, false>), float);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<float, true>),
                   float);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<double, true>),
                   double);

}  // namespace tensorflow
