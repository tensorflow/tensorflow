/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "third_party/eigen3/unsupported/Eigen/MatrixFunctions"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class MatrixExponentialOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixExponentialOp(OpKernelConstruction* context) : Base(context) {}

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) return;
    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix tmp = input;
    outputs->at(0) = tmp.exp();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixExponentialOp);
};

// Deprecated kernels (2018/08/21).
REGISTER_LINALG_OP("MatrixExponential", (MatrixExponentialOp<float>), float);
REGISTER_LINALG_OP("MatrixExponential", (MatrixExponentialOp<double>), double);
REGISTER_LINALG_OP("MatrixExponential", (MatrixExponentialOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("MatrixExponential", (MatrixExponentialOp<complex128>),
                   complex128);

}  // namespace tensorflow
