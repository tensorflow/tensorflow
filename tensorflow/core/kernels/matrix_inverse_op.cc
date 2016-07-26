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

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperation>
class MatrixInverseOp : public LinearAlgebraOp<Scalar, SupportsBatchOperation> {
 public:
  typedef LinearAlgebraOp<Scalar, SupportsBatchOperation> Base;

  explicit MatrixInverseOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      // By definition, an empty matrix's inverse is an empty matrix.
      return;
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition;
    if (adjoint_) {
      // TODO(rmlarsen): For Eigen 3.2, this creates a temporary copy.
      // Make sure to backport: https://bitbucket.org/eigen/eigen/commits/ \
      // bd2219a74c96dfe3f6bc2c23588749e36d2d8173
      lu_decomposition.compute(input.adjoint());
    } else {
      lu_decomposition.compute(input);
    }
    // TODO(rmlarsen): Add check based on condition number estimation.
    // PartialPivLU cannot give strong guarantees on invertibility, but
    // we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes, such as providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const Scalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > Scalar(0),
                errors::InvalidArgument("Input is not invertible."));
    outputs->at(0).noalias() = lu_decomposition.inverse();
  }

 private:
  bool adjoint_;

  TF_DISALLOW_COPY_AND_ASSIGN(MatrixInverseOp);
};

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float, false>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float, true>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double, true>),
                   double);

}  // namespace tensorflow
