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
#include "third_party/eigen3/Eigen/Eigenvalues"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {

template <class Scalar>
class SelfAdjointEigOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit SelfAdjointEigOp(OpKernelConstruction* context) : Base(context) {}

  using TensorShapes = typename Base::TensorShapes;
  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64 d = input_matrix_shapes[0].dim_size(0);
    return TensorShapes({TensorShape({d + 1, d})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const int64 rows = inputs[0].rows();
    if (rows == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }

    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        es(inputs[0]);
    OP_REQUIRES(context, es.info() == Eigen::Success,
                errors::InvalidArgument("Self Adjoint Eigen decomposition was"
                                        "not successful. "
                                        "The input might not be valid."));
    outputs->at(0).row(0) = es.eigenvalues().transpose();
    outputs->at(0).bottomRows(rows) = es.eigenvectors();
  }
};

REGISTER_LINALG_OP("SelfAdjointEig", (SelfAdjointEigOp<float>), float);
REGISTER_LINALG_OP("SelfAdjointEig", (SelfAdjointEigOp<double>), double);
REGISTER_LINALG_OP("BatchSelfAdjointEig", (SelfAdjointEigOp<float>), float);
REGISTER_LINALG_OP("BatchSelfAdjointEig", (SelfAdjointEigOp<double>), double);
}  // namespace tensorflow
